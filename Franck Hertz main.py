#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:09:38 2022

@author: harrytabb
"""

import numpy as np
import matplotlib.pyplot as plt

#importing data
filename1 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Franck Hertz/Mercury_readings/DataMercury3ms.csv'
filename2 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Franck Hertz/Mercury_readings/DataMercury4ms.csv'
filename3 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Franck Hertz/Mercury_readings/DataMercury5ms.csv'

#function to seperate collumns
def extractdata(filename):
    
    data = np.genfromtxt(filename, dtype=float, delimiter=',')
    voltage = data[:,0]
    current = data[:,1]
    return voltage, current, data

data1 = extractdata(filename1)[2]

#sorting data into 5 bins to fit polynomials to
p1_rows = np.unique(np.where((3.5 < data1) & (data1 < 5))[0])
p2_rows = np.unique(np.where((8 < data1) & (data1 < 10))[0])
p3_rows = np.unique(np.where((13 < data1) & (data1 < 15))[0])
p4_rows = np.unique(np.where((18 < data1) & (data1 < 20))[0])
p5_rows = np.unique(np.where((23 < data1) & (data1 < 25.5))[0])

#function to fit a polynomial to each data bin and calculate error on each coefficient
#Also finds the minima of each quadratic
def coeff(rows):
    data = data1[rows]
    voltage = data[:,0]
    current = data[:,1]
    fit_coeff = np.polyfit(voltage, current, deg=3, cov=True)
    
    a = fit_coeff[0][0]
    b = fit_coeff[0][1]
    c = fit_coeff[0][2]
    d = fit_coeff[0][3]
    
    error_array = fit_coeff[1].diagonal()
    
    fit_current = (a*voltage**3) + (b*voltage**2) + (c*voltage) + d
    deriv = np.array([3*a, 2*b, c])
    
    maxima = np.roots(deriv)
    print(3*a)
    print(2*b)
    print(c)
    
    
    return fit_current, voltage, maxima, (error_array)**0.5, 3*a, 2*b, c

#function to find error on minima of quadratic
def quad_error(a, a_error, b, b_error, c, c_error):
    term1 = 4 * a * c * ((a_error / a)**2 + (c_error / c)**2)**0.5 #4ac
    term2 = 2 * b**2 * (b_error / b) #b^2
    term3 = (term1**2 + term2**2)**0.5 #b^2 - 4ac
    term4 = 0.5 * (b**2 - (4*a*c))**0.5 * ((term3 / (b**2 - (4 * a * c)))) #(b^2 - 4ac)^1/2
    term5 = (b_error**2 + term4**2)**0.5 #numerator
    overall = ((-b + (b**2 - (4*a*c))**0.5) / (2 * a)) * ( (term5 / (-b + (b**2 - (4*a*c))**0.5))**2 + (a_error / 2 * a)**2 )**0.5
    return overall

#running each file through analysis
voltage1 = extractdata(filename1)[0]
current1 = extractdata(filename1)[1]

voltage2 = extractdata(filename2)[0]
current2 = extractdata(filename2)[1]

voltage3 = extractdata(filename3)[0]
current3 = extractdata(filename3)[1]

#printing results to terminal
p1_current = coeff(p1_rows)[0]
p1_voltage = coeff(p1_rows)[1]
p1_max = coeff(p1_rows)[2][0]
p1_error = quad_error(coeff(p1_rows)[4], (coeff(p1_rows)[3][0]), coeff(p1_rows)[5], (coeff(p1_rows)[3][1]), coeff(p1_rows)[6], (coeff(p1_rows)[3][2]))
print("First Maximum is at ({0:.2f} +/- {1:.2f})V".format(p1_max, p1_error))

p2_current = coeff(p2_rows)[0]
p2_voltage = coeff(p2_rows)[1]
p2_max = coeff(p2_rows)[2][0]
p2_error = quad_error(coeff(p2_rows)[4], coeff(p2_rows)[3][0], coeff(p2_rows)[5], coeff(p2_rows)[3][1], coeff(p2_rows)[6], coeff(p2_rows)[3][2])
print("Second Maximum is at ({0:.2f} +/- {1:.2f})V".format(p2_max, p2_error))

p3_current = coeff(p3_rows)[0]
p3_voltage = coeff(p3_rows)[1]
p3_max = coeff(p3_rows)[2][0]
p3_error = quad_error(coeff(p3_rows)[4], coeff(p3_rows)[3][0], coeff(p3_rows)[5], coeff(p3_rows)[3][1], coeff(p3_rows)[6], coeff(p3_rows)[3][2])
print("Third Maximum is at ({0:.2f} +/- {1:.2f})V".format(p3_max, p3_error))

p4_current = np.delete(coeff(p4_rows)[0], 0, axis=0)
p4_voltage = np.delete(coeff(p4_rows)[1],0, axis =0)
p4_max = coeff(p4_rows)[2][0]
p4_error = quad_error(coeff(p4_rows)[4], coeff(p4_rows)[3][0], coeff(p4_rows)[5], coeff(p4_rows)[3][1], coeff(p4_rows)[6], coeff(p4_rows)[3][2])
print("Fourth Maximum is at ({0:.2f} +/- {1:.2f})V".format(p4_max, p4_error))

p5_current = np.delete(coeff(p5_rows)[0], 0, axis=0)
p5_voltage = np.delete(coeff(p5_rows)[1], 0, axis=0)
p5_max = coeff(p5_rows)[2][1]
p5_error = quad_error(coeff(p5_rows)[4], coeff(p5_rows)[3][0], coeff(p5_rows)[5], coeff(p5_rows)[3][1], coeff(p5_rows)[6], coeff(p5_rows)[3][2])
print("Fifth Maximum is at ({0:.2f} +/- {1:.2f})V".format(p5_max, p5_error))

#plotting data along with minima
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.scatter(voltage1, current1, marker='.', linewidth=0.001)
ax.set_xlabel('Scan Voltage / V')
ax.set_ylabel('Scaled Current')
ax.set_title('Mercury at $U_1$ = 1.94 $U_3$ = 1.53  1')
ax.grid(which='both', alpha=0.5)
ax.minorticks_on()
plt.axvline(4.3, c='r',linewidth=0.6)
plt.text(3.4,0.6, '$V_1$={:.2f}V'.format(p1_max), rotation=90)
plt.axvline(9.15, c='r', linewidth=0.6)
plt.text(8.3,0.6, '$V_2$={:.2f}V'.format(p2_max), rotation=90)
plt.axvline(14.1, c='r',linewidth=0.6)
plt.text(13.1,0.6, '$V_3$={:.2f}V'.format(p3_max), rotation=90)
plt.axvline(19.2, c='r',linewidth=0.6)
plt.text(18.1,0.6, '$V_4$={:.2f}V'.format(p4_max), rotation=90)
plt.axvline(24.4, c='r',linewidth=0.6)
plt.text(23.3,0.66, '$V_5$={:.2f}V'.format(p5_max), rotation=90)
plt.plot(p1_voltage, p1_current, 'k')
plt.plot(p2_voltage, p2_current, 'k')
plt.plot(p3_voltage, p3_current, 'k')
plt.plot(p4_voltage, p4_current, 'k')
plt.plot(p5_voltage, p5_current, 'k')
plt.savefig('/Users/harrytabb/Desktop/Uni/Uni Year 2/Franck Hertz/Mercury_readings/fit.png', dpi=500)

# plt.figure()
# plt.plot(voltage2, current2, ',')
# plt.xlabel('Scan Voltage / V')
# plt.ylabel('Current')
# plt.title('Mercury at $U_1$ = 1.94 $U_3$ = 1.53  2')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(voltage3, current3, marker='.', linewidth=0.001)
# ax.set_xlabel('Scan Voltage / V')
# ax.set_ylabel('Scaled Current')
# ax.set_title('Mercury at $U_1$ = 1.94 $U_3$ = 1.53  3')
# ax.grid(which='both', alpha=0.5)
# ax.minorticks_on()
# plt.axvline(4.3, c='r',linewidth=0.6)
# plt.text(3.4,0.6, '$V_1$=4.3V', rotation=90)
# plt.axvline(9.2, c='r', linewidth=0.6)
# plt.text(8.3,0.6, '$V_2$=9.2V', rotation=90)
# plt.axvline(14.2, c='r',linewidth=0.6)
# plt.text(13.1,0.6, '$V_3$=14.2V', rotation=90)
# plt.axvline(19.3, c='r',linewidth=0.6)
# plt.text(18.1,0.6, '$V_4$=19.3V', rotation=90)
# plt.axvline(24.5, c='r',linewidth=0.6)
# plt.text(23.3,0.63, '$V_5$=24.5V', rotation=90)
# plt.savefig('/Users/harrytabb/Desktop/Uni/Uni Year 2/Franck Hertz/Mercury_readings/DataMercury3fig3.png', dpi=500)

