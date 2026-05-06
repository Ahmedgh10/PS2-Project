import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
print("Loading data...")
df = pd.read_csv('risk_students.csv')
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Socio_Economic_Status'] = df['Socio_Economic_Status'].map({'Low': 0, 'Medium': 1, 'High': 2})

X = df.drop('At_Risk', axis=1)
y = df['At_Risk']

# 2. Train-Test Split and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for inference
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to scaler.pkl")

# 3. Define the Best Model (ANN 1 Baseline)
class ANN_Baseline(nn.Module):
    def __init__(self, input_dim):
        super(ANN_Baseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# 4. Train the model precisely as in the notebook
print("Training model...")
input_dim = X_train_scaled.shape[1]
model = ANN_Baseline(input_dim)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)

num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Save the PyTorch model
torch.save(model.state_dict(), 'ann_model.pth')
print("Model saved to ann_model.pth")
