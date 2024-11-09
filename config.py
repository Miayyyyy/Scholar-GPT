API_URL_REDIRECT = {}
DEFAULT_WORKER_NUM = 3

THEME = "Default"
AVAIL_THEMES = ["Default", "Chuanhu-Small-and-Beautiful", "High-Contrast", "Gstaff/Xkcd", "NoCrypt/Miku"]

INIT_SYS_PROMPT = "Serve me as a writing and programming assistant."

CHATBOT_HEIGHT = 1115

CODE_HIGHLIGHT = True

LAYOUT = "LEFT-RIGHT"   # "LEFT-RIGHT"（左右布局） # "TOP-DOWN"（上下布局）
DARK_MODE = True
TIMEOUT_SECONDS = 30
WEB_PORT = -1

MAX_RETRY = 2

# 插件分类默认选项
DEFAULT_FN_GROUPS = ['对话', '编程', '学术', '智能体']

MULTI_QUERY_LLM_MODELS = "gpt-3.5-turbo&chatglm3"
CONCURRENT_COUNT = 100

AUTO_CLEAR_TXT = False
ADD_WAIFU = False

# [("username", "password"), ("username2", "password2"), ...]
AUTHENTICATION = []
# HTTPS 秘钥和证书（不需要修改）
SSL_KEYFILE = ""
SSL_CERTFILE = ""

TTS_TYPE = "EDGE_TTS" # EDGE_TTS / LOCAL_SOVITS_API / DISABLE
GPT_SOVITS_URL = ""
EDGE_TTS_VOICE = "zh-CN-XiaoxiaoNeural"

MATHPIX_APPID = ""
MATHPIX_APPKEY = ""

# pdf
GROBID_URLS = [
    "https://qingxu98-grobid.hf.space","https://qingxu98-grobid2.hf.space","https://qingxu98-grobid3.hf.space",
    "https://qingxu98-grobid4.hf.space","https://qingxu98-grobid5.hf.space", "https://qingxu98-grobid6.hf.space",
    "https://qingxu98-grobid7.hf.space", "https://qingxu98-grobid8.hf.space",
]
SEARXNG_URL = "https://cloud-1.agent-matrix.com/"
ALLOW_RESET_CONFIG = False
AUTOGEN_USE_DOCKER = False
PATH_PRIVATE_UPLOAD = "private_upload"
PATH_LOGGING = "gpt_log"
WHEN_TO_USE_PROXY = ["Download_LLM", "Download_Gradio_Theme", "Connect_Grobid",
                     "Warmup_Modules", "Nougat_Download", "AutoGen", "Connect_OpenAI_Embedding"]

