import { Redis } from '@upstash/redis'
import { createHash } from 'node:crypto'

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

/**
 * Retrieves all learned custom animals from KV.
 * Returns an empty array if KV is not configured or key is empty.
 */
export async function getCustomAnimals(includePending = false): Promise<CustomAnimalSignature[]> {
  if (!redis) {
    return [];
  }

  try {
    const [legacy, hash] = await Promise.all([
      redis.get<CustomAnimalSignature[]>(ANIMALS_KV_KEY),
      redis.hgetall<Record<string, CustomAnimalSignature>>(ANIMALS_HASH_KEY),
    ]);
    const merged = new Map<string, CustomAnimalSignature>();
    for (const animal of Array.isArray(legacy) ? legacy : []) merged.set(animal.id, animal);
    for (const animal of Object.values(hash || {})) merged.set(animal.id, animal);
    const arr = [...merged.values()];
    if (includePending) return arr;
    return arr.filter(a => a.status === "APPROVED");
  } catch (error) {
    console.error("[Akinator DB] Error fetching custom animals:", error);
    return [];
  }
}

/**
 * Saves a new custom animal to the KV database.
 */
export async function saveCustomAnimal(animal: CustomAnimalSignature): Promise<boolean> {
  if (!redis) {
    return false;
  }

  try {
    return (await redis.hsetnx(ANIMALS_HASH_KEY, animal.id, animal)) === 1;
  } catch (error) {
    console.error("[Akinator DB] Error saving custom animal:", error);
    return false;
  }
}

export async function claimPostGameToken(token: string): Promise<boolean> {
  const digest = createHash('sha256').update(token).digest('hex');
  if (redis) {
    const result = await redis.set(`akinator_reveal:${digest}`, '1', { nx: true, ex: 7200 });
    return result === 'OK';
  }

  const now = Date.now();
  for (const [key, expiresAt] of localRevealClaims) {
    if (expiresAt <= now) localRevealClaims.delete(key);
  }
  if (localRevealClaims.has(digest)) return false;
  localRevealClaims.set(digest, now + 2 * 60 * 60_000);
  return true;
}

export async function releasePostGameToken(token: string): Promise<void> {
  const digest = createHash('sha256').update(token).digest('hex');
  if (redis) {
    await redis.del(`akinator_reveal:${digest}`);
    return;
  }
  localRevealClaims.delete(digest);
}

export function hasCustomAnimalStore() {
  return redis !== null;
}
