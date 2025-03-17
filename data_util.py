import matplotlib.pyplot as plt
import seaborn as sns

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



def bp_data(df):
    healthy = df[df["blood pressure class"] == "healthy"]
    elevated = df[df["blood pressure class"] == "elevated"]
    stage1 = df[df["blood pressure class"] == "stage 1 hypertension"]
    stage2 = df[df["blood pressure class"] == "stage 2 hypertension"]
    crisis = df[df["blood pressure class"] == "hypertension crisis"]
    return [len(healthy), len(elevated), len(stage1), len(stage2), len(crisis)]


def plot_bmi_vs_bp(df):
    normal_range = df[df["bmi class"] == "normal range"]
    over_weight = df[df["bmi class"] == "over-weight"]
    obese1 = df[df["bmi class"] == "obese (class I)"]
    obese2 = df[df["bmi class"] == "obese (class II)"]
    obese3 = df[df["bmi class"] == "obese (class III)"]

    bp_labels = ["healthy", "elevated", "stage 1 hypertension", "stage 2 hypertension", "hypertension crisis"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15), dpi=120)

    def plot(ax, df, title):
        ax.pie(bp_data(df), labels=bp_labels, autopct="%.1f%%")
        ax.set_title(title)

    plot(axs[0, 0], normal_range, "Blood Pressure Distribution for Normal Range")
    plot(axs[0, 1], over_weight, "Blood Pressure Distribution for Over Weight")
    plot(axs[1, 0], obese1, "Blood Pressure Distribution for Obese (Class I)")
    plot(axs[1, 1], obese2, "Blood Pressure Distribution for Obese (Class II)")
    plot(axs[2, 0], obese3, "Blood Pressure Distribution for Obese (Class III)")

    axs[2, 1].axis('off')


