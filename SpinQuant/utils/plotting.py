import matplotlib.pyplot as plt
import torch
import numpy as np

# plotting
# hidden_states = torch.load("./17_hidden_fused_basis_hadamard_blockdiag.pt")
# x = (
#     torch.linspace(0, int(hidden_states.shape[-1]) - 1, int(hidden_states.shape[-1]))
#     .cpu()
#     .numpy()
# )
# y = hidden_states.mean(dim=(0, 1)).float().cpu()
# # y = hidden_states[0, 0].float().cpu()
# plt.figure(figsize=(10, 4))
# plt.plot(x, y, color="blue", label="Sine Wave")
# # Fill the area under the curve with custom shading color
# plt.fill_between(x, y, color="lightblue", alpha=0.5)
# # Adding labels and title
# plt.xlabel("Channel #")
# # plt.ylabel("Value")
# plt.title("MHA Input")

# # Show legend
# # plt.legend()

# # Add grid lines with customization
# plt.grid(color="gray", linestyle="--", linewidth=0.5)
# plt.show()
# # Display the plot
# plt.savefig("17_hidden_fused_basis_hadamard_blockdiag.png")
# plt.clf()
####################################
weight = torch.load("./temp_q_8.pt")
x = torch.linspace(0, int(weight.shape[1]) - 1, int(weight.shape[1])).cpu().numpy()
# y = weight[16, :].float().cpu().detach().numpy()
y = weight.abs().mean(0).float().cpu().detach().numpy()
plt.plot(x, y, color="blue", label="Sine Wave")
# Fill the area under the curve with custom shading color
plt.fill_between(x, y, color="lightblue", alpha=0.5)
# Adding labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Qproj_17_baseline")

# Show legend
plt.legend()

# Add grid lines with customization
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.show()
# Display the plot
plt.savefig("temp.png")
plt.clf()

# ####################################
# weight = torch.load("./17_kproj_baseline.pt")
# x = torch.linspace(0, 1, int(weight.shape[0])).cpu().numpy()

# y = weight.mean(dim=(1,)).float().cpu().detach().numpy()
# plt.plot(x, y, color="blue", label="Sine Wave")
# # Fill the area under the curve with custom shading color
# plt.fill_between(x, y, color="lightblue", alpha=0.5)
# # Adding labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Kproj_17_baseline")

# # Show legend
# plt.legend()

# # Add grid lines with customization
# plt.grid(color="gray", linestyle="--", linewidth=0.5)
# plt.show()
# # Display the plot
# plt.savefig("wk_17_baseline.png")
# plt.clf()

# ####################################
# weight = torch.load("./17_vproj_baseline.pt")
# x = torch.linspace(0, 1, int(weight.shape[0])).cpu().numpy()
# y = weight.mean(dim=(1,)).float().cpu().detach().numpy()
# plt.plot(x, y, color="blue", label="Sine Wave")
# # Fill the area under the curve with custom shading color
# plt.fill_between(x, y, color="lightblue", alpha=0.5)
# # Adding labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Vproj_17_baseline")

# # Show legend
# plt.legend()

# # Add grid lines with customization
# plt.grid(color="gray", linestyle="--", linewidth=0.5)
# plt.show()
# # Display the plot
# plt.savefig("wv_17_baseline.png")
# plt.clf()

# ####################################
# weight = torch.load("./17_oproj_baseline.pt")
# x = torch.linspace(0, 1, int(weight.shape[0])).cpu().numpy()

# y = weight.mean(dim=(1)).float().cpu().detach().numpy()
# plt.plot(x, y, color="blue", label="Sine Wave")
# # Fill the area under the curve with custom shading color
# plt.fill_between(x, y, color="lightblue", alpha=0.5)
# # Adding labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Oproj_17_baseline")

# # Show legend
# plt.legend()

# # Add grid lines with customization
# plt.grid(color="gray", linestyle="--", linewidth=0.5)
# plt.show()
# # Display the plot
# plt.savefig("wo_17_baseline.png")
# plt.clf()
