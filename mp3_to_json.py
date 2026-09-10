import whisper
import json
import os

model=whisper.load_model("large-v2")

audios= os.listdir("Audios")

for audio in audios:
    # print(audio)
    if("_" in audio):
        number=audio.split("_")[0]
        title=audio.split("_")[1][:-4]
        print(number,title)
        # result=model.transcribe(audio="Audios/output.mp3",language="hi",task="translate",word_timestamps=False)
        result=model.transcribe(audio=f"Audios/{audio}",language="hi",task="translate",word_timestamps=False)
        chunks=[]
    for segment in result["segments"]:
        chunks.append({"number":number,"title":title,"start":segment["start"],"end":segment["end"],"text":segment["text"]})

    chunks_with_metadata={"chunks":chunks,"text":result["text"]} #text hmne isliye liya because we want to know whether iss text me question ka answer h ya nhi

    with open(f"jsons/{audio}.json","w") as f:
        json.dump(chunks_with_metadata,f)