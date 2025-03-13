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