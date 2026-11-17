import { Redis } from '@upstash/redis'
import { createHash } from 'node:crypto'
import { QUESTIONS } from './akinator-kb'
import { isExcludedAnimalName } from './akinator-scope'

// Initialize Redis if env vars are present (prevents crash on local if not set up yet)
const redis = process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN
  ? new Redis({
      url: process.env.KV_REST_API_URL,
      token: process.env.KV_REST_API_TOKEN,
    })
  : null;

// Type for the learned animals
export type CustomAnimalSignature = {
  id: string;
  name: { en: string; tr: string; ar: string };
  traits: Record<string, boolean>;
  status: "PENDING_REVIEW" | "APPROVED" | "REJECTED";
  submittedAt: string;
};

const ANIMALS_KV_KEY = "akinator_custom_animals_v1";
const ANIMALS_HASH_KEY = "akinator_custom_animals_v2";
const localRevealClaims = new Map<string, number>();
const localRateLimits = new Map<string, { count: number; expiresAt: number }>();
const MAX_LOCAL_ENTRIES = 10_000;

export function validateCustomAnimal(value: unknown): CustomAnimalSignature | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const row = value as Partial<CustomAnimalSignature>;
  if (typeof row.id !== 'string' || !/^[a-z0-9][a-z0-9-]{0,79}$/.test(row.id)) return null;
  if (!row.name || !['en', 'tr', 'ar'].every(lang => {
    const name = row.name?.[lang as 'en' | 'tr' | 'ar'];
    return typeof name === 'string' && name.trim().length > 0 && name.length <= 80 && !/[\u0000-\u001f]/.test(name);
  })) return null;
  if (!['APPROVED', 'REJECTED', 'PENDING_REVIEW'].includes(row.status ?? '')) return null;
  if (typeof row.submittedAt !== 'string' || !Number.isFinite(Date.parse(row.submittedAt))) return null;
  if (!row.traits || typeof row.traits !== 'object' || Array.isArray(row.traits)) return null;
  const traits = Object.fromEntries(QUESTIONS.filter(q => typeof row.traits?.[q.id] === 'boolean').map(q => [q.id, row.traits![q.id]]));
  return { id: row.id, name: { en: row.name.en.trim(), tr: row.name.tr.trim(), ar: row.name.ar.trim() }, traits, status: row.status!, submittedAt: row.submittedAt };
}

export function customAnimalApprovalIssues(animal: Pick<CustomAnimalSignature, 'traits' | 'name'>) {
  return {
    missing_traits: QUESTIONS.filter(q => typeof animal.traits[q.id] !== 'boolean').map(q => q.id),
    out_of_scope: Object.values(animal.name).some(isExcludedAnimalName),
  };
}

export async function consumeAkinatorRateLimit(key: string, limit: number, seconds: number): Promise<boolean> {
  if (redis) {
    try {
      const bucket = `akinator_rate:${key}:${Math.floor(Date.now() / (seconds * 1000))}`;
      const count = await redis.incr(bucket);
      if (count === 1) await redis.expire(bucket, seconds * 2);
      return count <= limit;
    } catch { /* Bounded per-instance fallback when shared storage is unavailable. */ }
  }
  const now = Date.now();
  for (const [key, entry] of localRateLimits) if (entry.expiresAt <= now) localRateLimits.delete(key);
  const current = localRateLimits.get(key);
  if (!current && localRateLimits.size >= MAX_LOCAL_ENTRIES) return false;
  const entry = current ?? { count: 0, expiresAt: now + seconds * 1000 };
  entry.count++;
  localRateLimits.set(key, entry);
  return entry.count <= limit;
}

// The engine calls getCustomAnimals() on every single game turn to rebuild
// its runtime catalog. The approved-animal list only changes when an admin
// moderates a submission, so a short TTL cache avoids a Redis round-trip
// (two commands: GET + HGETALL) on every question/answer in every session.
const CUSTOM_ANIMALS_CACHE_TTL_MS = 60_000;
let customAnimalsCache: { all: CustomAnimalSignature[]; fetchedAt: number } | null = null;

async function fetchAllCustomAnimals(): Promise<CustomAnimalSignature[]> {
  if (!redis) return [];
  if (customAnimalsCache && Date.now() - customAnimalsCache.fetchedAt < CUSTOM_ANIMALS_CACHE_TTL_MS) {
    return customAnimalsCache.all;
  }

  try {
    const [legacy, hash] = await Promise.all([
      redis.get<CustomAnimalSignature[]>(ANIMALS_KV_KEY),
      redis.hgetall<Record<string, CustomAnimalSignature>>(ANIMALS_HASH_KEY),
    ]);
    const merged = new Map<string, CustomAnimalSignature>();
    for (const raw of [...(Array.isArray(legacy) ? legacy : []), ...Object.values(hash || {})]) {
      const animal = validateCustomAnimal(raw);
      if (animal) merged.set(animal.id, animal);
    }
    const all = [...merged.values()];
    customAnimalsCache = { all, fetchedAt: Date.now() };
    return all;
  } catch (error) {
    console.error("[Akinator DB] Error fetching custom animals:", error);
    return customAnimalsCache?.all ?? [];
  }
}

/**
 * Retrieves all learned custom animals from KV.
 * Returns an empty array if KV is not configured or key is empty.
 */
export async function getCustomAnimals(includePending = false): Promise<CustomAnimalSignature[]> {
  const all = await fetchAllCustomAnimals();
  if (includePending) return all;
  return all.filter(a => a.status === "APPROVED");
}

/**
 * Saves a new custom animal to the KV database.
 */
export async function saveCustomAnimal(animal: CustomAnimalSignature): Promise<boolean> {
  if (!redis || !validateCustomAnimal(animal)) {
    return false;
  }
  
  try {
    const saved = (await redis.hsetnx(ANIMALS_HASH_KEY, animal.id, animal)) === 1;
    if (saved) customAnimalsCache = null;
    return saved;
  } catch (error) {
    console.error("[Akinator DB] Error saving custom animal:", error);
    return false;
  }
}

/**
 * Updates the status of an existing custom animal.
 */
export async function updateCustomAnimalStatus(id: string, status: "APPROVED" | "REJECTED" | "PENDING_REVIEW"): Promise<boolean> {
  return updateCustomAnimalProfile(id, { status });
}

/** Full trait replacement: absent keys remain unreviewed, never false. */
export async function updateCustomAnimalProfile(id: string, changes: { status?: CustomAnimalSignature['status']; traits?: Record<string, boolean> }): Promise<boolean> {
  if (!redis) return false;
  try {
    if (!/^[a-z0-9][a-z0-9-]{0,79}$/.test(id) || (changes.status !== undefined && !['APPROVED', 'REJECTED', 'PENDING_REVIEW'].includes(changes.status))) return false;
    const allowed = new Set(QUESTIONS.map(q => q.id));
    if (changes.traits && Object.entries(changes.traits).some(([key, value]) => !allowed.has(key) || typeof value !== 'boolean')) return false;
    // Promote a legacy record to the hash on first moderation; the legacy
    // array remains preserved, and hash precedence makes the status effective.
    const animal = (await fetchAllCustomAnimals()).find(animal => animal.id === id);
    if (!animal) return false;
    const updated = { ...animal, ...changes };
    const issues = customAnimalApprovalIssues(updated);
    if (updated.status === 'APPROVED' && (issues.out_of_scope || issues.missing_traits.length)) return false;
    // Overwrite the existing field (hsetnx would fail if it exists, so we use hset)
    await redis.hset(ANIMALS_HASH_KEY, { [id]: updated });
    customAnimalsCache = null;
    return true;
  } catch (error) {
    console.error("[Akinator DB] Error updating custom animal status:", error);
    return false;
  }
}

/**
 * Retires a custom animal while preserving its record for moderation history.
 */
export async function deleteCustomAnimal(id: string): Promise<boolean> {
  return updateCustomAnimalStatus(id, 'REJECTED');
}

export async function claimPostGameToken(token: string, gameId?: string): Promise<boolean> {
  if (!/^v[34]\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token)) return false;
  if (gameId !== undefined && !/^[a-f0-9]{32}$/.test(gameId)) return false;
  const digest = createHash('sha256').update(gameId ?? token).digest('hex');
  if (redis) {
    try {
      const result = await redis.set(`akinator_reveal:${digest}`, '1', { nx: true, ex: 7200 });
      return result === 'OK';
    } catch { /* Verification remains available without storage. */ }
  }

  const now = Date.now();
  for (const [key, expiresAt] of localRevealClaims) {
    if (expiresAt <= now) localRevealClaims.delete(key);
  }
  if (localRevealClaims.has(digest)) return false;
  if (localRevealClaims.size >= MAX_LOCAL_ENTRIES) return false;
  localRevealClaims.set(digest, now + 2 * 60 * 60_000);
  return true;
}

export async function releasePostGameToken(token: string, gameId?: string): Promise<void> {
  const digest = createHash('sha256').update(gameId ?? token).digest('hex');
  if (redis) {
    await redis.del(`akinator_reveal:${digest}`).catch(() => undefined);
  }
  localRevealClaims.delete(digest);
}

export function hasCustomAnimalStore() {
  return redis !== null;
}
