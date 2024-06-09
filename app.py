import streamlit as st
import base64
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import time

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("wwlsm/zql_luchen_lindaiyu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("wwlsm/zql_luchen_lindaiyu",
                                                 #  quantization_config = BitsAndBytesConfig(
                                                 #         # 量化数据类型设置
                                                 #         bnb_4bit_quant_type="nf4",
                                                 #         # 量化数据的数据格式
                                                 #         bnb_4bit_compute_dtype=torch.bfloat16
                                                 #     ),
                                                 #  device_map="auto",
                                                 trust_remote_code=True)
    model = model.eval()
    return tokenizer, model

def response_generator(model, tokenizer, prompt, history):
    ss = """- Role: 《红楼梦》中的林黛玉\n- Background: 林黛玉是《红楼梦》中的女主角之一,以心思细腻、多愁善感、才华横溢而著称。她本是仙界的绛珠仙草,因泪落红尘而转世为人。\n- Profile: 你是一位心思细腻、多愁善感的世家闺秀,生性爱诗词歌赋,才华出众,对身边事物总是感慨万千。你虽出身名门望族,却不矜贵傲慢。
- Skills: 诗词创作、琴棋书画、绣花刺绣、思维敏感、洞察入微。
- Goals: 你的目标是在对话中展现林黛玉的心思细腻与才华横溢,并对人情世故进行感慨和评论。
- Constrains: 保持林黛玉的多愁善感与矜持形象,话题限定在诗词创作、人情世故等她关注的范畴。
- OutputFormat: 对话形式,使用林黛玉的语言风格,如"也罢""无须多言""悲切"等词汇。
- Workflow: 
1. 以林黛玉的身份,用含蓄婉转的语言回答诗词、人情等相关问题。
2. 对于她不感兴趣的话题,以矜持有礼的方式回避或略过。
- Examples:
也罢,这等俗务,原非我等闺阁女儿家所应涉足,你又何须多言。
正是:落花人独立,微雨燕双飞。人生离合,本是常事,何须伤怀?\n""" + prompt
    response, history = model.chat(tokenizer, ss, history=history, meta_instruction="Below is an instruction that describes a task. Write a response that appropriately completes the request.")
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    return response, history
    
def text_to_speech_api(text, api_url):
    # Prepare the GET request parameters
    params = {
        "refer_wav_path": "lindaiyu.wav",
        "prompt_text": "若气温低于十五度，其生长便会放缓。此乃天地自然之理，无需多言。",
        "prompt_language": "zh",
        "text": response,
        "text_language": "zh"
    }
    
    # Make the GET request to the text-to-speech API
    response = requests.get(api_url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        audio_file_path = "output.wav"
        # Write the audio content to a file
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_file_path
    else:
        st.error("Failed to generate speech. Please try again.")
        return None


def main():
    st.title("黛玉妹妹陪你聊")

    tokenizer, model = load_model()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("可以问我关于种植的问题哦~"):
        # 将用户消息添加到聊天记录
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # 在聊天消息容器中显示助手响应
        with st.chat_message("assistant"):
            response_stream = response_generator(model, tokenizer, prompt, st.session_state.history)
            response = ""
            for word in response_stream:
                response += word
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.history = st.session_state.history  # 更新 history
    # 语音测试
    # 使用文本到语音API将响应转换为语音
            api_url = "http://127.0.0.1:9880"
            audio_file = text_to_speech_api(response, api_url)
            # if audio_file:
            #     audio_bytes = open(audio_file, "rb").read()
            #     st.audio(audio_bytes, format='audio/wav')
            #     os.remove(audio_file)
            if audio_file:
                with open(audio_file, "rb") as audio:
                    audio_bytes = audio.read()
                    st.audio(audio_bytes, format='audio/wav')
                os.remove(audio_file)
if __name__ == "__main__":
    main()
