import pandas as pd

HOTPOT_QA_CQG_DATASET_PATH = "./data_cqg_split.json"
HOTPOT_QA_TF_DATASET_PATH = "./hotpot_qa_dataset_pred_using_T5.csv"
SQUAD_TRAIN_T5_DATASET_PATH = "./squad_train_dataset_pred_t5.csv"
SQUAD_VAL_T5_DATASET_PATH = "./squad_val_dataset_pred_t5.csv"

if __name__ == "__main__":
    hotpot_qa_cgq_df = pd.read_json(
        HOTPOT_QA_CQG_DATASET_PATH, orient="records")
    hotpot_qa_cgq_df.rename(columns={"gold": "question", "fact": "context",
                                     "ans": "answer", "pred": "generated_qn"}, inplace=True)
    hotpot_qa_tf_df = pd.read_csv(HOTPOT_QA_TF_DATASET_PATH)
    squad_train_tf_df = pd.read_csv(SQUAD_TRAIN_T5_DATASET_PATH)
    squad_val_tf_df = pd.read_csv(SQUAD_VAL_T5_DATASET_PATH)
    merged_df = pd.concat([hotpot_qa_cgq_df, hotpot_qa_tf_df,
                          squad_train_tf_df, squad_val_tf_df], ignore_index=True, axis=0)
    merged_df.to_csv("merged_dataset.csv", index=False)
