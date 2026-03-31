# A full interface needs to be implemented.

from spellcheck import MLSpellChecker

# from train import train_and_save_model

# train_and_save_model(text_corpus=input(), external_dict_path="wordlist.dic")

if __name__ == "__main__":
    try:
        checker = MLSpellChecker(
            model_path="language_model.json",
            top_n=20,
            cutoff=0.4,
            sim_weight=3,
            context_weight=2,
            debug=True,
        )

        cau_sai = input()
        print(f"\nCâu gốc: {cau_sai}")

        cau_dung = checker.correct_sentence(cau_sai)
        print(f"Câu sửa: {cau_dung}")

    except FileNotFoundError:
        print("language_model.json not found!")
