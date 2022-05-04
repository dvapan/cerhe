import scipy as sc
import more_itertools as mit



length = 4                      # Длина теплообменника         [м]
total_time = 300                      # Время работы теплообменника  [с]
rball = 0.01                    # Радиус однгого шара засыпки  [м]
rbckfill = 2                    # Радиус засыпки               [м]
fi = 0.4                        # Пористость                   [доля]
MN = 4186.                      # Коэффициент перевода в килокаллории

# Расчет объема засыпки [м^3]
vbckfill = sc.pi * rbckfill**2 * (1 - fi)

# Расчет количества шаров [шт]
cball = vbckfill/((4*sc.pi/3)*rball**3)

# Расчет удельной площади теплообмена [м^2]
surf_spec = cball * 4 * sc.pi * rball ** 2

# Расчет площади живого сечения для прохождения теплоносителя через засыпку  [м^2]
fgib = sc.pi*fi*rbckfill**2

# Расчет эквивалентного диаметра засыпки (для расчета теплообмена) [м]
dekb=(4/3)*(fi/(1-fi))*rball

TG = 1000

xreg,treg = 3,3
max_reg = xreg*treg
max_poly_degree = 3
ppr = 10                        # Точек на регион

accs = {
        "eq_cer_heat": 1,#0.001,
        "eq_cer_cool": 1,#0.001,
        "eq_gas_heat": 1,#10,
        "eq_gas_cool": 1,#10,
        "eq_sur_heat": 1,#0.001,
        "eq_sur_cool": 1,#0.001,
               "temp": 1,#1,
}
