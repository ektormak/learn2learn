import matplotlib.pyplot as plt
from learn_by_grad_desc_by_grad_desc.train import train_optimizer


if __name__ == "__main__":
    n_epochs = 350
    training_losses = train_optimizer(n_epochs=n_epochs)
    plt.plot(range(n_epochs), training_losses)
    plt.show()
