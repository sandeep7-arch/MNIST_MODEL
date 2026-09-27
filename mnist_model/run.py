import torch
import numpy as np
from PIL import Image
from main import mnist

img = Image.open("1007.png").convert("L").resize((28, 28))

tensor = torch.from_numpy(np.array(img, dtype=np.float32)).flatten() / 255.0


model = mnist() 

model.load_state_dict(torch.load('final_model.pth', weights_only=True))
model.eval()
y_pred = model.forward(tensor)

prediction = y_pred.argmax(dim=1).item()
print("Predicted Digit:", prediction) 
