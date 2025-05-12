from utils import *
from encoder import *
from decoder import *
from dataloader import *
import multiprocessing

def main():
    chamfer_dist = ChamferDistance()

    # device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    encoder = Encoder().to(device)
    decoder = Decoder(input_dim=2).to(device)

    optimiser = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = 1e-3)

    train_loader = getDataloader(data_dir='ModelNet10', split='train', batch_size=2)
    test_loader = getDataloader(data_dir='ModelNet10', split='test', batch_size=1)

    num_epochs = 100

    best_val_loss = float('inf')
    model_save_path = 'best_model.pth'

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for batch,_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch = batch.to(device).float()

            optimiser.zero_grad()

            emb = encoder(batch)
            recon = decoder(emb)

            loss = chamfer_dist(recon, batch)

            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}")

        encoder.eval()
        decoder.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch,_ in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                batch = batch.to(device).float()


                emb = encoder(batch)
                recon = decoder(emb)

                loss = chamfer_dist(recon, batch)
                val_loss += loss.item()
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_val_loss:.6f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save({
                'epoch': epoch+1,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimiser': optimiser.state_dict(),

            }, model_save_path)

            print(f"Model saved with best val loss {best_val_loss}")


if __name__=='__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
