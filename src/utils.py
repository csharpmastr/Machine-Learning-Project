import seaborn as sns
import matplotlib.pyplot as plt

# function to visualize distribution before and after sampling
def vizSampling(sampled_data, pre_sampled_data, sampling_method=''):
    plt.figure(figsize=(12, 6))
    
    # Plot the distribution before SMOTE sampling
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x='CLASS', data=pre_sampled_data)
    for p in ax1.patches:
        ax1.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() + 0.3,
        '{:.0f}'.format(p.get_height()),
        ha='center',
        va='bottom'
        )
    plt.title(f'Distribution of Target Variable (Before {sampling_method})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Plot the distribution after SMOTE sampling
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x='CLASS', data=sampled_data)
    for p in ax2.patches:
        ax2.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() + 0.3,
        '{:.0f}'.format(p.get_height()),
        ha='center',
        va='bottom'
        )
        
    plt.title(f'Distribution of Target Variable (After {sampling_method})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
