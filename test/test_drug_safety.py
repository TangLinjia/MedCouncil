from src.tools.drug_safety_check import drug_safety_check


print("\n=== TEST 1: CURRENT DRUG INTERACTION ===")

result = drug_safety_check.invoke(
    {
        "proposed_medications": [
            "Nitroglycerin 0.3 mg sublingual"
        ],
        "current_medications": [
            "sildenafil"
        ],
        "allergies": [],
    }
)

print(result)


print("\n=== TEST 2: CLASS-LEVEL INTERACTION ===")

result = drug_safety_check.invoke(
    {
        "proposed_medications": [
            "Ibuprofen 400 mg"
        ],
        "current_medications": [
            "warfarin"
        ],
        "allergies": [],
    }
)

print(result)


print("\n=== TEST 3: ALLERGY ===")

result = drug_safety_check.invoke(
    {
        "proposed_medications": [
            "Amoxicillin 500 mg"
        ],
        "current_medications": [],
        "allergies": [
            "penicillin allergy"
        ],
    }
)

print(result)


print("\n=== TEST 4: PROPOSED VS PROPOSED ===")

result = drug_safety_check.invoke(
    {
        "proposed_medications": [
            "Nitroglycerin",
            "Sildenafil"
        ],
        "current_medications": [],
        "allergies": [],
    }
)

print(result)
