import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("下载NLTK资源...")
nltk.download('vader_lexicon')
nltk.download('punkt')
print("下载完成!") 