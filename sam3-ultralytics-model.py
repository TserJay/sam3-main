from ultralytics.models.sam import SAM3SemanticPredictor
import time


# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image(r"/mnt/pengyi-sam3/sam3-main/assets/images/car3.png")


results = predictor(text=["person", "bus", "car"])

begin_time=time.time()*1000
# Query with multiple text prompts
results = predictor(text=["person", "bus", "car"])
end_time=time.time()*1000
print(f"推理时间：{end_time-begin_time}")
print(results)

begin_time=time.time()*1000
# Works with descriptive phrases
results = predictor(text=["car in ground", "person with blue cloth"])
end_time=time.time()*1000
print(f"推理时间：{end_time-begin_time}")
print(results)



begin_time=time.time()*1000
# Query with a single concept
results = predictor(text=["a person"])
end_time=time.time()*1000
print(f"推理时间：{end_time-begin_time}")
print(results)

