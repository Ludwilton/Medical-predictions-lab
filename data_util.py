import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def bmi_calc(weight, height):
    return (weight / ((height/100)**2))


def bmi_class_assign(bmi):
    if bmi < 24.9:
        return "normal range"
    elif bmi < 29.9:
        return "over-weight"
    elif bmi < 34.9:
        return "obese (class I)"
    elif bmi < 39.9:
        return "obese (class II)"
    else:
        return "obese (class III)"
    
def bp_class_assign(systolic, diastolic):
    if systolic >= 180 or diastolic >= 120:
        return "hypertension crisis"
    elif systolic >= 140 or diastolic >= 90:
        return "stage 2 hypertension"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "stage 1 hypertension"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "elevated"
    elif systolic < 120 and diastolic < 80:
        return "healthy"
    
    
'''    
def bp_class_assign(systolic, diastolic): # denna orsakade fel på över 180,120
    if systolic < 120 and diastolic < 80:
        return "healthy"
    elif systolic < 129 and diastolic < 80:
        return "elevated"
    elif systolic < 139 and diastolic < 89:
        return "stage 1 hypertension"
    elif systolic <= 179 and diastolic <= 119:
        return "stage 2 hypertension"
    elif systolic >= 180 and diastolic >= 120:
        return "hypertension crisis"
'''



def plot_positive_vs_negative_bp(df): # plots positive vs negative blood pressure classes
    positive = df[df["cardio"] == 1] 
    negative = df[df["cardio"] == 0]
    
    positive_bp = positive["blood pressure class"].value_counts()
    negative_bp = negative["blood pressure class"].value_counts()
    
    
    plt.figure(figsize=(12, 6))
    bp_classes = [
        "healthy",
        "elevated",
        "stage 1 hypertension",
        "stage 2 hypertension",
        "hypertension crisis"
    ]

    x = range(len(bp_classes))                                                                # ---* start
    width = 0.40                                                                              # *slight help from gpt:
    pos_values = [positive_bp.get(cls, 0) for cls in bp_classes]                              # "i need assistance aligning these bars next to each other
    neg_values = [negative_bp.get(cls, 0) for cls in bp_classes]                              #  -while maintaining correct colors for each class"
    plt.bar([i - width/2 for i in x], pos_values, width, label='Heart Disease Positive', color='#1f77b4')     #  reason: had trouble with bars overlapping each other
    plt.bar([i + width/2 for i in x], neg_values, width, label='Heart Disease Negative', color='#ff7f0e')     #  ---* end

    plt.xlabel('Blood Pressure Class')
    plt.ylabel('Count')
    plt.title('Blood pressure class Distribution: Positive vs Negative for heart Disease')
    plt.xticks(x, bp_classes)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # this is basically the same as the above BP classes, 
    # could've easily been a dynamic function but ill refactor this when
    # time isnt a pressure point
def plot_positive_vs_negative_bmi(df): # plot positive vs negative bmi
    positive = df[df["cardio"] == 1] 
    negative = df[df["cardio"] == 0]

    positive_bmi = positive["bmi class"].value_counts()
    negative_bmi = negative["bmi class"].value_counts()

    plt.figure(figsize=(12, 6))
    
    bmi_classes = [
        "normal range",
        "over-weight",
        "obese (class I)",
        "obese (class II)",
        "obese (class III)"
    ]
    
    x = range(len(bmi_classes))                                                     # ---* start
    width = 0.40                                                                    # *slight help from gpt:
    pos_values = [positive_bmi.get(cls, 0) for cls in bmi_classes]                  # "i need assistance aligning these bars next to each other
    neg_values = [negative_bmi.get(cls, 0) for cls in bmi_classes]                  #  -while maintaining correct colors for each class"
    plt.bar([i - width/2 for i in x], pos_values, width, label='Heart Disease Positive', color='#1f77b4')  #  reason: had trouble with bars overlapping each other
    plt.bar([i + width/2 for i in x], neg_values, width, label='Heart Disease Negative', color='#ff7f0e')  #  ---* end
    
    plt.xlabel('BMI Class')
    plt.ylabel('Count')
    plt.title('BMI Distribution: Heart Disease Positive vs Negative')
    plt.xticks(x, bmi_classes)
    plt.legend()
    plt.tight_layout()
    plt.show()









def bp_data(df):
    healthy = df[df["blood pressure class"] == "healthy"]
    elevated = df[df["blood pressure class"] == "elevated"]
    stage1 = df[df["blood pressure class"] == "stage 1 hypertension"]
    stage2 = df[df["blood pressure class"] == "stage 2 hypertension"]
    crisis = df[df["blood pressure class"] == "hypertension crisis"]
    return [len(healthy), len(elevated), len(stage1), len(stage2), len(crisis)]


def plot_bmi_vs_bp(df):
    colors = sns.color_palette("Set2", 5)  
    
    normal_range = df[df["bmi class"] == "normal range"]
    over_weight = df[df["bmi class"] == "over-weight"]
    obese1 = df[df["bmi class"] == "obese (class I)"]
    obese2 = df[df["bmi class"] == "obese (class II)"]
    obese3 = df[df["bmi class"] == "obese (class III)"]

    bp_labels = ["healthy", "elevated", "stage 1 hypertension", "stage 2 hypertension", "hypertension crisis"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15), dpi=120)
    fig.suptitle("BMI vs Blood pressure", fontsize=25)
    def plot(ax, df, title):
        ax.pie(bp_data(df), labels=bp_labels, autopct="%.1f%%", colors=colors)
        ax.set_title(title)
        ax.set()

    plot(axs[0, 1], normal_range, "BMI: Normal Range")
    plot(axs[1, 0], over_weight, "BMI: Over Weight")
    plot(axs[1, 1], obese1, "BMI: Obese (Class I)")
    plot(axs[2, 0], obese2, "BMI: Obese (Class II)")
    plot(axs[2, 1], obese3, "BMI: Obese (Class III)")

    patches = [mpatches.Patch(color=colors[i], label=bp_labels[i]) for i in range(len(bp_labels))]
    axs[0, 0].legend(handles=patches, loc='center', fontsize='x-large',
                     title='Blood Pressure Classes', title_fontsize='xx-large',
                     bbox_to_anchor=(0, 0, 1, 1), bbox_transform=axs[0, 0].transAxes)
    axs[0, 0].axis('off')
