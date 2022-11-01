import json
import argparse
from tqdm import tqdm
import os


def convert_data_format(data, **kwargs):

    top_answers = min(kwargs.get('top_answers', 1), 3)

    new_data = []
    for data_instance in tqdm(data['data']):
        
        for content in data_instance.get("paragraphs", []):
            
            context = content.get("context", "")
            for instance in content.get("qas", []):

                question = instance.get("question", "")  
                ID = instance.get("id", "")        
                for answer in instance.get("answers", [])[:top_answers]:
                    ans = answer.get("text", "")

                    new_data.append({
                        'id': ID,
                        'annotation1': {
                            'toks': context
                        },
                        'annotation2': {
                            'toks': question
                        },
                        'annotation3': {
                            'toks': ans
                        }
                    })

    return new_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output file')
    args = vars(parser.parse_args())

    input_file = args.get('input', '')
    if os.path.exists(input_file):
        with open(input_file) as file:
            data = json.load(file)
    else:
        raise FileNotFoundError(f'{input_file} does not exists')

    # Considering only the first of the available answers. Change top answers to maximum 3 as the dataset consists of max 3 answers
    new_data = convert_data_format(data, top_answers=1)

    output_file = args.get('output', '')    
    print(f'Writing data to file {output_file}')
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)

