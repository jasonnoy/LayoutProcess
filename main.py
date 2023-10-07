from CoyoOcrProcessor import CoyoOcrProcessor
import json
from tqdm import tqdm


def process_coyo_ocr(file_path, output_path):
    ocr_processor = CoyoOcrProcessor()
    with open(file_path, 'r') as f, open(output_path, 'w', encoding='utf-8') as f2:
        for line in tqdm(f):
            data = json.loads(line)
            ocr_data = data['paddle_ocr']
            res_data = []
            for page_data in ocr_data:
                p_data = ocr_processor(page_data)
                res_data.append(p_data)
            data['paddle_ocr'] = res_data
            f2.write(json.dumps(data, ensure_ascii=False) + '\n')
    f2.close()


def main():
    file_path = "/share/hwy/data/coyo_paddleocr/part-00000/000000.ocr.jsonl"
    output_path = "/share/jjh/data/coyo_paddleocr_processed/part-00000/000000.ocr.jsonl"
    process_coyo_ocr(file_path, output_path)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
