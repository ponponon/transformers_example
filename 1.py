from transformers import AutoTokenizer, AutoModel
from loguru import logger
import time

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     "THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')

model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).to('mps')


model = model.eval()


tt = []

with open('2.txt', 'r', encoding='utf-8') as file:

    for line in file.readlines():

        text = f"""
        「{line.strip().rsplit('.')[0]}」，对前面的句子做实体识别，要求包含人物、地点、时间等信息,以 json 格式输出
        """.strip()

        s = time.time()
        response, history = model.chat(tokenizer, text, history=[])
        e = time.time()

        tt.append(e-s)

        logger.debug(text)
        logger.debug(response)
        logger.debug(f'耗时: {round(e-s,3)} 秒')
        print('-----------------------------------------------')

        with open('output-30.txt', 'a', encoding='utf-8') as out_file:
            out_file.write(text)
            out_file.write('\n')
            out_file.write(response)
            out_file.write('\n-----------------------------------------------')
            out_file.write('\n')
            out_file.write('\n')
            out_file.write('\n')

print(f'总耗时: {sum(tt)}秒')
print(f'平均单个耗时: {sum(tt)/len(tt)}秒')