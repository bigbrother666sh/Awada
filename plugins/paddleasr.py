import os, pilk
import datetime
from paddlespeech.cli.asr.infer import ASRExecutor
#from paddlespeech.cli.text.infer import TextExecutor


def pcm2wav(input_pcm, out_wav, sr):
    cmd = f"ffmpeg -y -f s16le -ar {sr} -ac 1 -i {input_pcm} {out_wav}"
    r = os.system(cmd)
    if r == 0:
        return True
    else:
        return False


def silk2wav(input_silk, out_wav, sr, out_pcm="temp.pcm"):
    duration = pilk.decode(input_silk, pcm=out_pcm, pcm_rate=sr)
    if pcm2wav(out_pcm, out_wav, sr):
        return out_wav, duration
    else:
        return None


asr_model = ASRExecutor()
#text_punc = TextExecutor()

def asr(talker: str, input_silk: str, cache_url: str) -> str:
    timestmp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    outwav = os.path.join(cache_url, f"asr_{talker}{timestmp}.wav")
    outpcm = os.path.join(cache_url, f"pcm_{talker}{timestmp}.pcm")
    trans_result = silk2wav(input_silk, outwav, sr=16000, out_pcm=outpcm)
    if trans_result:
        out_wav, _ = trans_result
        asr_result = asr_model(out_wav, force_yes=True)
    else:
        return "######"
    return asr_result
