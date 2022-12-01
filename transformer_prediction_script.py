#!pip install transformers==4.24.0 sentencepiece rouge

import re
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

DATASET_PATH = "Encoder_Decoder/datasets/merged_dataset.csv"
RESULTS_SAVE_PATH = "Encoder_Decoder/datasets/prediction_results.csv"


tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")


def generate_question(answer: str, context: str, max_length: int = 64) -> str:
    input_text = "answer: %s  context: %s " % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'],
                            max_length=max_length)

    return tokenizer.decode(output[0])


def read_dataset(path: str) -> pd.DataFrame:
    dataset: pd.DataFrame = pd.read_csv(path)
    dataset = dataset.sample(frac=1)
    print(f'\nTotal number of records: {len(dataset)}')
    dataset = dataset.dropna()
    print(
        f'Total number of records after dropping NaN values: {len(dataset)}\n')
    return dataset


def calculate_scores(dataset: pd.DataFrame, csv_save_path: str) -> None:
    prediction_results = {"pred_1": [], "pred_2": [], "pred_3": []}
    rouge_scores = {"pred_1": 0, "pred_2": 0, "pred_3": 0}
    rouge = Rouge()

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        for key in prediction_results.keys():
            try:
                if key == "pred_1":
                    res = generate_question(
                        row['answer'], f"{row['context']} {row['generated_qn']}")
                elif key == "pred_2":
                    res = generate_question(
                        row['generated_qn'], row['context'])
                else:
                    res = generate_question(
                        f"{row['generated_qn']} {row['answer']}", row['context'])
                parsed_res = re.search('question: (.+?)</s>', res).group(1)
                prediction_results[key].append(parsed_res)
                rouge_scores[key] += rouge.get_scores([parsed_res], [row['question']])[
                    0]['rouge-l']['f']
            except Exception as e:
                print(str(e))
                prediction_results[key].append(None)

    dataset['pred_1'], dataset['pred_2'], dataset['pred_3'] = prediction_results.values()
    dataset = dataset.dropna()
    dataset.to_csv(csv_save_path)
    for col in ['question', 'pred_1', 'pred_2', 'pred_3']:
        if col == 'question':
            dataset[col] = dataset[col].apply(lambda x: [word_tokenize(x)])
        else:
            dataset[col] = dataset[col].apply(lambda x: word_tokenize(x))

    print("Results:\n")
    for key in rouge_scores.keys():
        print(f"\nRouge Score ({key}): ", rouge_scores[key]/len(dataset))
        print(f"BLEU-1 {key}:", corpus_bleu(
            dataset['question'].to_list(), dataset[key].to_list(), weights=(1, 0, 0, 0)))
        print(f"BLEU-2 {key}:", corpus_bleu(dataset['question'].to_list(
        ), dataset[key].to_list(), weights=(0.5, 0.5, 0, 0)))
        print(f"BLEU-3 {key}:", corpus_bleu(dataset['question'].to_list(
        ), dataset[key].to_list(), weights=(0.33, 0.33, 0.33, 0)))
        print(f"BLEU-4 {key}:", corpus_bleu(dataset['question'].to_list(
        ), dataset[key].to_list(), weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == "__main__":
    dataset: pd.DataFrame = read_dataset(DATASET_PATH)
    calculate_scores(dataset, RESULTS_SAVE_PATH)
