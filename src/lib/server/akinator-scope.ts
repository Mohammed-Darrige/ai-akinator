/** Product scope restrictions, also applied to community profiles. */
export function isExcludedAnimalName(input: string): boolean {
    const name = input.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "").replace(/ı/g, "i");
    const excluded = ["monkey", "ape", "chimpanzee", "gorilla", "orangutan", "lemur", "baboon", "macaque", "gibbon", "dinosaur", "tyrannosaurus", "triceratops", "stegosaurus", "brachiosaurus", "velociraptor", "dodo", "mammoth", "saber-toothed", "maymun", "sempanze", "goril", "orangutan", "dinozor", "mamut", "قرد", "شمبانزي", "غوريلا", "ديناصور", "ماموث"];
    return excluded.some(word => new RegExp(`(^|[^\\p{L}])${word}($|[^\\p{L}])`, "u").test(name));
}
