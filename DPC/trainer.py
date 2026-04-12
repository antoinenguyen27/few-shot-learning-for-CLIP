import torch
import torch.nn.functional as F

class DPCTrainer:
    def __init__(self, lr=1e-3, epochs=10, temperature=1.0, alpha=0.8):
        self.lr = lr
        self.epochs = epochs
        self.T = temperature
        self.alpha = alpha

    def train(self, model, train_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in train_loader:
                images = batch["image"]
                labels = batch["label"]

                logits, reference_logits = model(images)

                # Cross-entropy loss
                ce_loss = F.cross_entropy(logits, labels)

                # KL divergence loss
                log_probs = F.log_softmax(logits / self.T, dim=1)
                ref_probs = F.softmax(reference_logits / self.T, dim=1)

                kl_loss = F.kl_div(log_probs, ref_probs, reduction="batchmean") * (self.T ** 2)

                # Combined loss
                loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        return model