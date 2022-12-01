import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_embeddings import SentenceEmbeddings

trained_model_path = './model/'
trained_tokenizer_path = './tokenizer/'
squad_train_dataset_path = './datasets/squad_train.csv'
save_train_dataset_path = './datasets/train_dataset.csv'
squad_val_dataset_path = './datasets/squad_validation.csv'
save_val_dataset_path = './datasets/val_dataset.csv'
cqg_dataset_path = './datasets/data_cqg_split.json'
save_cqg_dataset_path = './datasets/cqg_dataset.csv'

torch.cuda.empty_cache()


class QuestionGeneration:

    def __init__(self, model_dir=None):
        self.model = T5ForConditionalGeneration.from_pretrained(
            trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        ).to(self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=3,
            num_return_sequences=1
        )
        question_list = []
        for output in outputs:
            question = self.tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            question_list.append(
                {'question': question, 'answer': answer, 'context': context})
        return question_list


if __name__ == "__main__":
    QG = QuestionGeneration()
    SE = SentenceEmbeddings()
    # SQuAD
    for (read_path, write_path) in zip([squad_train_dataset_path, squad_val_dataset_path], [save_train_dataset_path, save_val_dataset_path]):
        df = pd.read_csv(read_path)
        generated_qn = []
        for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                qa_pair_list = QG.generate(row['answer'], row['context'])
                most_similar = SE.get_most_similar(
                    row['context'], qa_pair_list)
                generated_qn.append(most_similar['question'])
            except Exception as e:
                print(e)
                generated_qn.append(None)
        df['generated_qn'] = generated_qn
        df.to_csv(write_path, index=False)
    # HotpotQA
    df = pd.read_json(cqg_dataset_path, orient="records")
    df.drop('pred', axis=1, inplace=True)
    df.rename(columns={"gold": "question", "fact": "context",
              "ans": "answer"}, inplace=True)
    generated_qn = []
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            qa_pair_list = QG.generate(row['answer'], row['context'])
            most_similar = SE.get_most_similar(row['context'], qa_pair_list)
            generated_qn.append(most_similar['question'])
        except Exception as e:
            print(e)
            generated_qn.append(None)
    df['generated_qn'] = generated_qn
    df.to_csv(save_cqg_dataset_path, index=False)
