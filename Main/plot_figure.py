import numpy as np
import matplotlib.pyplot as plt


hd_result = np.loadtxt("plot_data/hd_result.csv", delimiter=",")
md_result = np.loadtxt("plot_data/md_result.csv", delimiter=",")

types = ["No Reranker (MIPS)", "BGE Reranker", "JINA Reranker", "GTE Reranker", "ListCR"]
length = 2417
gd_result = 2417 - hd_result - md_result
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

for ty, dum in zip(types, hd_result):
    axs[2].plot(range(1, 6), np.array(dum) * 100 / length, marker='o', label=ty)
x_y_font = 18
title_font = 20
para_size = 20
# axs[0].legend(title='Reranker')
axs[2].set_xlabel("Rank of the Document", fontsize=x_y_font)
axs[2].set_ylabel("Percentage (%)", fontsize=x_y_font)
axs[2].set_title("(c) Percentage of HD at Top-5 Rank", fontsize=title_font)
axs[2].grid(True, linestyle='--', alpha=0.7)
axs[2].tick_params(axis='both', which='major', labelsize=para_size)
axs[2].set_xticks(np.arange(1, 6, 1))
axs[2].set_yticks(np.arange(7, 35, 5))
axs[2].set_xlim(0.9, 5.1)

for ty, dum in zip(types, md_result):
    axs[1].plot(range(1, 6), np.array(dum) * 100 / length, marker='o', label=ty)

axs[1].set_xlabel("Rank of the Document", fontsize=x_y_font)
axs[1].set_ylabel("Percentage (%)", fontsize=x_y_font)
axs[1].set_title("(b) Percentage of MD at Top-5 Rank", fontsize=title_font)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].tick_params(axis='both', which='major', labelsize=para_size)
axs[1].set_xticks(np.arange(1, 6, 1))
axs[1].set_xlim(0.9, 5.1)
axs[1].set_ylim(0, 70.1)
axs[1].set_yticks(np.arange(0, 71, 10))
for ty, dum in zip(types, gd_result):
    axs[0].plot(range(1, 6), np.array(dum) * 100 / length, marker='o', label=ty)

axs[2].legend(fontsize = 14, loc='upper right')
axs[0].set_xlabel("Rank of the Document", fontsize=x_y_font)
axs[0].set_ylabel("Percentage (%)", fontsize=x_y_font)
axs[0].set_title("(a) Percentage of GD at Top-5 Rank", fontsize=title_font)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].tick_params(axis='both', which='major', labelsize=para_size)
axs[0].set_xticks(np.arange(1, 6, 1))
axs[0].set_xlim(0.9, 5.1)
axs[0].set_ylim(20, 90.1)
axs[0].set_yticks(np.arange(20, 91, 10))
plt.tight_layout()
plt.savefig("figure/empirical study.png")
plt.show()

def plot_bar_chart(values, colors, filename, tick_font_size=12, label_font_size=10):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    labels = ['GD', 'HD', 'MD']
    
    fig, ax = plt.subplots(figsize=(4, 2))
    bars = ax.bar(labels, values, color=colors, edgecolor='none')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval)}%', va='bottom', ha='center', fontsize=label_font_size)
    
    ax.set_ylim(0, 80)
    
    ax.yaxis.set_visible(False)
    
    plt.xticks(fontsize=tick_font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(f'figure/{filename}.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)
# This section does not depict the experimental results, but rather the schematic diagram in the main image.
plot_bar_chart([79,12,9], ['#E2F0D9', '#FBE5D6', '#DEEBF7'], 'mips_distribution', tick_font_size=20, label_font_size=20)
plot_bar_chart([62,9,29], ['#E2F0D9', '#FBE5D6', '#DEEBF7'], 'reranker_distribution', tick_font_size=20, label_font_size=20)


data = np.genfromtxt("plot_data/reasoning.csv",delimiter=",",skip_footer=0)
dataset_names = ['Musique', 'NQ', 'MultiHopQA','PQA-L']
setting_names = ['our method', '-R(x)']


bar_width = 0.35
index = np.arange(len(dataset_names))

fig, ax = plt.subplots(figsize = (8, 4))
bars1 = ax.bar(index - bar_width/2, data[:, 0], bar_width, label=setting_names[0], color='#00B050')
bars2 = ax.bar(index + bar_width/2, data[:, 1], bar_width, label=setting_names[1], color='#FE2300')

ax.set_xlabel('Datasets', fontsize = para_size)
ax.set_ylabel('Number', fontsize = para_size)
ax.set_title('Hesitation Frequency', fontsize = title_font)
ax.set_xticks(index)
ax.set_xticklabels(dataset_names)
ax.legend(fontsize = para_size - 2)
ax.tick_params(axis='both', which='major', labelsize=para_size)
ax.set_ylim(0,4.5)
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=para_size - 2
                    )

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.savefig("figure/reasoning.png")
plt.show()