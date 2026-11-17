import matplotlib.pyplot as plt
import numpy as np
import os

# Set dark theme style
plt.style.use('dark_background')
fig_bgcolor = '#0e1117'
ax_bgcolor = '#161b22'

# ---------------------------------------------------------------------
# Figure 1: Fused LIF Neuron Benchmark
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=fig_bgcolor)
for ax in (ax1, ax2):
    ax.set_facecolor(ax_bgcolor)
    ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

sizes = ['100K', '1M', '10M']
pytorch_lif = [0.0719, 0.0792, 1.2565]
triton_lif = [0.0812, 0.0824, 1.0495]
speedups_lif = [0.89, 0.96, 1.20]

x = np.arange(len(sizes))
width = 0.35

rects1 = ax1.bar(x - width/2, pytorch_lif, width, label='PyTorch Eager', color='#ff6b6b')
rects2 = ax1.bar(x + width/2, triton_lif, width, label='Triton Fused LIF', color='#4ecdc4')

ax1.set_ylabel('Execution Time per Step (ms)', fontsize=12, fontweight='bold', color='white')
ax1.set_xlabel('Tensor Elements (N)', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Fused LIF Neuron Latency (RTX 5060)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(sizes, fontweight='bold')
ax1.legend(frameon=True, facecolor=ax_bgcolor, edgecolor='none')

# Speedup plot
ax2.plot(sizes, speedups_lif, marker='o', linewidth=3, markersize=8, color='#00e676')
for i, txt in enumerate(speedups_lif):
    ax2.annotate(f"{txt:.2f}x", (sizes[i], speedups_lif[i]), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='#00e676', fontsize=11)

ax2.set_ylabel('Speedup Factor vs PyTorch', fontsize=12, fontweight='bold', color='white')
ax2.set_xlabel('Tensor Elements (N)', fontsize=12, fontweight='bold', color='white')
ax2.set_title('Fused LIF Speedup Factor', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(0.5, 1.5)

plt.tight_layout()
plt.savefig('figures/lif_speedup_benchmark.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Regenerated figures/lif_speedup_benchmark.png")

# ---------------------------------------------------------------------
# Figure 2: Fused STDP Plasticity Benchmark
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=fig_bgcolor)
for ax in (ax1, ax2):
    ax.set_facecolor(ax_bgcolor)
    ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

dims = ['512x512\n(262K syn)', '2048x2048\n(4.19M syn)', '4096x4096\n(16.78M syn)']
pytorch_stdp = [0.0703, 0.6098, 3.2252]
triton_stdp = [0.0262, 0.0979, 0.6283]
speedups_stdp = [2.68, 6.23, 5.13]

x = np.arange(len(dims))

rects1 = ax1.bar(x - width/2, pytorch_stdp, width, label='PyTorch Outer Product', color='#ff6b6b')
rects2 = ax1.bar(x + width/2, triton_stdp, width, label='Triton 2D Fused STDP', color='#4ecdc4')

ax1.set_ylabel('Weight Update Latency (ms)', fontsize=12, fontweight='bold', color='white')
ax1.set_xlabel('Synaptic Weight Matrix Dimensions', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Fused STDP Plasticity Latency (RTX 5060)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(dims, fontweight='bold')
ax1.legend(frameon=True, facecolor=ax_bgcolor, edgecolor='none')

# Speedup plot with margin offset to avoid y-axis/label collision
ax2.plot(dims, speedups_stdp, marker='s', linewidth=3, markersize=8, color='#00e676')
for i, txt in enumerate(speedups_stdp):
    ax2.annotate(f"{txt:.2f}x Speedup", (dims[i], speedups_stdp[i]), textcoords="offset points", 
                 xytext=(0,12), ha='center', fontweight='bold', color='#00e676', fontsize=11)

ax2.set_ylabel('Speedup Factor vs PyTorch', fontsize=12, fontweight='bold', color='white')
ax2.set_xlabel('Synaptic Weight Matrix Dimensions', fontsize=12, fontweight='bold', color='white')
ax2.set_title('Fused STDP Plasticity Speedup', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(1.0, 8.0)

plt.tight_layout()
plt.savefig('figures/stdp_speedup_benchmark.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Regenerated figures/stdp_speedup_benchmark.png")

# ---------------------------------------------------------------------
# Figure 3: Fused Causal Spiking Attention Benchmark
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=fig_bgcolor)
for ax in (ax1, ax2):
    ax.set_facecolor(ax_bgcolor)
    ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

seq_lens = ['T = 256', 'T = 1024', 'T = 2048']
pytorch_attn = [0.1259, 0.2879, 1.0801]
triton_attn = [0.0345, 0.0500, 0.1476]
speedups_attn = [3.65, 5.76, 7.32]

x = np.arange(len(seq_lens))

rects1 = ax1.bar(x - width/2, pytorch_attn, width, label='PyTorch Causal Attention', color='#ff6b6b')
rects2 = ax1.bar(x + width/2, triton_attn, width, label='Triton Spiking Attention', color='#4ecdc4')

ax1.set_ylabel('Attention Latency (ms)', fontsize=12, fontweight='bold', color='white')
ax1.set_xlabel('Sequence Length (T)', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Fused Causal Spiking Attention Latency', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(seq_lens, fontweight='bold')
ax1.legend(frameon=True, facecolor=ax_bgcolor, edgecolor='none')

# Speedup plot
ax2.plot(seq_lens, speedups_attn, marker='^', linewidth=3, markersize=8, color='#00e676')
for i, txt in enumerate(speedups_attn):
    ax2.annotate(f"{txt:.2f}x Speedup", (seq_lens[i], speedups_attn[i]), textcoords="offset points", 
                 xytext=(0,12), ha='center', fontweight='bold', color='#00e676', fontsize=11)

ax2.set_ylabel('Speedup Factor vs PyTorch', fontsize=12, fontweight='bold', color='white')
ax2.set_xlabel('Sequence Length (T)', fontsize=12, fontweight='bold', color='white')
ax2.set_title('Spiking Attention Speedup Factor', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(2.0, 9.0)

plt.tight_layout()
plt.savefig('figures/spiking_attention_speedup.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Regenerated figures/spiking_attention_speedup.png")
