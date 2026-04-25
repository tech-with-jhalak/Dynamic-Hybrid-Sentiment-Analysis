from hybrid_model import hybrid_predict

text = input("Enter text: ").strip()

if not text:
    print("❌ Please enter valid text!")
else:
    result = hybrid_predict(text)

    print("\n--- RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")