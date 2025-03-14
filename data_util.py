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