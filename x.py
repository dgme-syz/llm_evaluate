from bleurt import score

checkpoint = "/home/nfs05/shenyz/bleurt/bleurt/BLEURT-20"
references = ["This is a test.", "整个20世纪60年代，布雷兹尼克为约翰·F·肯尼迪工作，担任他的顾问，之后又在林登·B·约翰逊政府中任职。"]
candidates = ["This is the test.", "20 世纪 60 年代，布热津斯基先是为约翰·福特·肯尼迪效力，担任他的顾问，后来又为林登·贝恩斯·约翰逊政府工作。"]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
# assert isinstance(scores, list) and len(scores) == 1
print(scores)