import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    
def add_custom_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #ff69b4;
            text-align: center;
        }
        .user-input {
            font-size: 18px;
        }
        .response {
            font-size: 18px;
            color: #4b0082;
        }
        .stTextInput {
            border: 2px solid #ff69b4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    add_custom_css()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("https://oss.laf.dev/t62x87-imgs/11111.png", use_column_width=True)  
    with col2:
        st.write("<div class='main-title'>黛玉妹妹陪你聊</div>", unsafe_allow_html=True)
    with col3:
        st.write("") 
    
    tokenizer, model = load_model()
    
    history = []
    
    user_input = st.text_input("User:")
    
    if user_input:
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
正是:落花人独立,微雨燕双飞。人生离合,本是常事,何须伤怀?\n""" + user_input
        response, history = model.chat(tokenizer, ss, history=history, meta_instruction="Below is an instruction that describes a task. Write a response that appropriately completes the request.")
        st.write("Assistant:", response)

if __name__ == "__main__":
    main()
