import json
import random
from typing import List, Dict, Tuple

# Phonetic building blocks for random name generation
CONSONANTS = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'y', 'z']
VOWELS = ['a', 'e', 'i', 'o', 'u']
INDIAN_CONSONANT_CLUSTERS = ['sh', 'th', 'ch', 'kh', 'gh', 'bh', 'dh', 'ph', 'pr', 'kr', 'tr', 'sr', 'br', 'gr']
CITY_PREFIXES = ['Ban', 'Chen', 'Hyd', 'Mum', 'Del', 'Kol', 'Pun', 'Jal', 'Vai', 'Nag', 'Sur', 'Ind', 'Ahm', 'Koch']
CITY_SUFFIXES = ['pur', 'bad', 'nai', 'ore', 'i', 'abad', 'kata', 'ar', 'gaon', 'puram', 'palli', 'giri']
DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "rediffmail", "protonmail", "mail", "email", "inbox"]

def generate_random_name() -> str:
    """Generate a random pronounceable name"""
    length = random.randint(2, 4)  # 2-4 syllables
    name = ""
    
    for i in range(length):
        if random.random() < 0.3 and i > 0:
            name += random.choice(INDIAN_CONSONANT_CLUSTERS)
        else:
            name += random.choice(CONSONANTS)
        name += random.choice(VOWELS)
        if random.random() < 0.3:  # Sometimes add ending consonant
            name += random.choice(['n', 'm', 'h', 'l', 'r', 's', 'sh', 't'])
    
    return name.capitalize()

def generate_random_city() -> str:
    """Generate a random city name"""
    if random.random() < 0.7:
        return random.choice(CITY_PREFIXES) + random.choice(CITY_SUFFIXES)
    else:
        # Fully random
        return generate_random_name() + random.choice(CITY_SUFFIXES)

def generate_credit_card() -> str:
    """Generate a random valid-format credit card number"""
    # Generate random but realistic card patterns
    prefixes = {
        'visa': ['4'],
        'mastercard': ['51', '52', '53', '54', '55'],
        'amex': ['34', '37'],
        'discover': ['6011', '65']
    }
    
    card_type = random.choice(list(prefixes.keys()))
    prefix = random.choice(prefixes[card_type])
    
    if card_type == 'amex':
        # Amex is 15 digits
        remaining = 15 - len(prefix)
        digits = prefix + ''.join([str(random.randint(0, 9)) for _ in range(remaining)])
        # Format: 4-6-5
        card = f"{digits[:4]} {digits[4:10]} {digits[10:]}"
    else:
        # Others are 16 digits
        remaining = 16 - len(prefix)
        digits = prefix + ''.join([str(random.randint(0, 9)) for _ in range(remaining)])
        # Format: 4-4-4-4
        card = f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"
    
    return card

def generate_phone() -> str:
    """Generate phone number in STT-style (spelled out)"""
    digits = [random.randint(0, 9) for _ in range(10)]
    
    # Various STT patterns
    patterns = [
        lambda d: " ".join([num_to_word(x) for x in d]),
        lambda d: " ".join([str(x) for x in d]),
        lambda d: " ".join([num_to_word(x) if random.random() > 0.3 else str(x) for x in d]),
        lambda d: "".join([num_to_word(x) + " " if i % 2 == 0 else str(x) + " " for i, x in enumerate(d)]).strip(),
    ]
    
    return random.choice(patterns)(digits)

def num_to_word(n: int) -> str:
    """Convert digit to word"""
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return words[n]

def generate_email(first: str, last: str) -> str:
    """Generate email in STT-style"""
    domain = random.choice(DOMAINS)
    
    patterns = [
        f"{first.lower()} dot {last.lower()} at {domain} dot com",
        f"{first.lower()} underscore {last.lower()} at {domain} dot com",
        f"{first.lower()}{random.randint(1, 99)} at {domain} dot com",
        f"{first.lower()}{last.lower()} at {domain} dot com",
        f"{first[0].lower()} dot {last.lower()} at {domain} dot com",
    ]
    
    return random.choice(patterns)

def find_entity_span(text: str, entity: str) -> Tuple[int, int]:
    """Find start and end position of entity in text"""
    start = text.find(entity)
    if start == -1:
        return -1, -1
    return start, start + len(entity)

def generate_example() -> Dict:
    """Generate a single training example"""
    first_name = generate_random_name()
    last_name = generate_random_name()
    city = generate_random_city()
    
    templates = [
        # Credit card + email
        lambda: {
            "parts": [
                f"my credit card number is {generate_credit_card()}",
                f"and my email is {generate_email(first_name, last_name)}"
            ],
            "entities": ["CREDIT_CARD", "EMAIL"]
        },
        # Phone + city
        lambda: {
            "parts": [
                f"call me on {generate_phone()}",
                f"i live in {city.lower()}"
            ],
            "entities": ["PHONE", "CITY"]
        },
        # Name + phone
        lambda: {
            "parts": [
                f"this is {first_name} {last_name}",
                f"my number is {generate_phone()}"
            ],
            "entities": ["PERSON_NAME", "PHONE"]
        },
        # Email + phone + city
        lambda: {
            "parts": [
                f"you can reach me at {generate_email(first_name, last_name)}",
                f"or call {generate_phone()}",
                f"i am from {city.lower()}"
            ],
            "entities": ["EMAIL", "PHONE", "CITY"]
        },
        # Credit card + name
        lambda: {
            "parts": [
                f"card holder name is {first_name} {last_name}",
                f"card number {generate_credit_card()}"
            ],
            "entities": ["PERSON_NAME", "CREDIT_CARD"]
        },
        # Phone only
        lambda: {
            "parts": [
                f"my phone number is {generate_phone()}"
            ],
            "entities": ["PHONE"]
        },
        # Email only
        lambda: {
            "parts": [
                f"send it to {generate_email(first_name, last_name)}"
            ],
            "entities": ["EMAIL"]
        },
        # Complex multi-entity
        lambda: {
            "parts": [
                f"hi i am {first_name} {last_name} from {city.lower()}",
                f"email me at {generate_email(first_name, last_name)}",
                f"or call {generate_phone()}"
            ],
            "entities": ["PERSON_NAME", "CITY", "EMAIL", "PHONE"]
        },
    ]
    
    template = random.choice(templates)()
    text = " ".join(template["parts"])
    
    # Find entity positions
    entities = []
    temp_text = text
    
    for entity_type in template["entities"]:
        if entity_type == "CREDIT_CARD":
            # Find credit card pattern
            words = text.split()
            for i, word in enumerate(words):
                if word.isdigit() or (len(word) == 4 and word.replace(" ", "").isdigit()):
                    # Look for credit card pattern
                    potential = " ".join(words[i:i+7])
                    if len([w for w in potential.split() if w.isdigit()]) >= 4:
                        start = text.find(potential)
                        entities.append({
                            "start": start,
                            "end": start + len(potential),
                            "label": "CREDIT_CARD"
                        })
                        break
        
        elif entity_type == "PHONE":
            # Find phone number (sequence of digit words)
            words = text.split()
            phone_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            for i, word in enumerate(words):
                if word in phone_words or word.isdigit():
                    # Found start of phone
                    j = i
                    while j < len(words) and (words[j] in phone_words or words[j].isdigit()):
                        j += 1
                    if j - i >= 8:  # At least 8 digits for phone
                        phone_text = " ".join(words[i:j])
                        start = text.find(phone_text)
                        entities.append({
                            "start": start,
                            "end": start + len(phone_text),
                            "label": "PHONE"
                        })
                        break
        
        elif entity_type == "EMAIL":
            # Find email pattern (contains "at" and "dot")
            if " at " in text and " dot com" in text:
                at_pos = text.find(" at ")
                dot_pos = text.find(" dot com", at_pos)
                # Find start of email (before "at")
                before_at = text[:at_pos].split()
                start_idx = text.rfind(before_at[-1])
                entities.append({
                    "start": start_idx,
                    "end": dot_pos + 8,
                    "label": "EMAIL"
                })
        
        elif entity_type == "PERSON_NAME":
            # Find first + last name pattern
            name = f"{first_name} {last_name}"
            start = text.lower().find(name.lower())
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(name),
                    "label": "PERSON_NAME"
                })
        
        elif entity_type == "CITY":
            start = text.lower().find(city.lower())
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(city.lower()),
                    "label": "CITY"
                })
    
    # Sort entities by start position
    entities.sort(key=lambda x: x["start"])
    
    return {
        "id": f"utt_{random.randint(1000, 9999)}",
        "text": text,
        "entities": entities
    }

def generate_dataset(num_examples: int, output_file: str):
    """Generate dataset and write to JSONL file"""
    examples = []
    for _ in range(num_examples):
        example = generate_example()
        examples.append(example)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_examples} examples in {output_file}")

if __name__ == "__main__":
    # Generate training set (1000 examples)
    train_size = 1000
    generate_dataset(train_size, "pii_ner_assignment/data/train.jsonl")
    
    # Generate dev set (200 examples)
    dev_size = 200
    generate_dataset(dev_size, "pii_ner_assignment/data/dev.jsonl")
    
    print(f"\nDataset generation complete!")
    print(f"Training examples: {train_size}")
    print(f"Dev examples: {dev_size}")