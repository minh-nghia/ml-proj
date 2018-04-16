from tkinter import *
import tensorflow as tf
import numpy as np
import random
import json
import matplotlib
import pprint
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure


import json

with open('country_convert.json') as f:
    countries = json.load(f)
with open('zone_convert.json') as f:
    zones = json.load(f)


class App(object):

    def __init__(self, master, sess):

        self.tf_session = sess

        frame = Frame(master)
        frame.grid()
        frame.winfo_toplevel().title('Anomaly Detective')


        frame_left = Frame(frame)
        frame_left.grid(column=0, row=0)

        frame_middle = Frame(frame)
        frame_middle.grid(column=1, row=0)

        frame_right = Frame(frame)
        frame_right.grid(column=2, row=0)

        Label(frame_left, text='Resource type').grid(column=0, row=0, sticky=W)
        self.resources = {
            'Instance': 1, 'Storage Bucket': 2, 'API': 3, 'Backend Service': 4,
            'Disk': 5, 'Instance Group': 6, 'Instance Group Manager': 7
        }
        self.resource_var = StringVar(frame_left, value=list(self.resources.keys())[0])
        self.resource_dropdown = OptionMenu(frame_left, self.resource_var, *tuple(self.resources.keys()))
        self.resource_dropdown.config(width=30)
        self.resource_dropdown.grid(column=1, row=0, sticky=EW)
        Button(frame_left, text='Random', command=self.random_resource).grid(column=3, row=0, sticky=W)

        frame2 = Frame(frame)
        frame2.grid()

        Label(frame_left, text='Request IP').grid(column=0, row=1, sticky=W)
        self.request_ip_max = 65
        self.request_ip_var = StringVar(frame_left)
        self.request_ip_var.set('IP Address 1')
        self.request_ip_dropdown = OptionMenu(frame_left, self.request_ip_var,
                                              *tuple(['IP Address ' + str(i) for i in range(1, self.request_ip_max + 1)]))
        self.request_ip_dropdown.grid(column=1, row=1, sticky=EW)
        Button(frame_left, text='Random', command=self.random_request_ip).grid(column=3, row=1, sticky=W)

        self.operations = {
            'ACCESS': 1, 'MODIFICATION': 2, 'PLACEMENT': 3
        }
        Label(frame_left, text='Operation').grid(column=0, row=2, sticky=W)
        self.operation_var = StringVar(frame_left, value=list(self.operations.keys())[0])
        self.operation_dropdown = OptionMenu(frame_left, self.operation_var, *tuple(self.operations.keys()))
        self.operation_dropdown.grid(column=1, row=2, sticky='EW')
        Button(frame_left, text='Random', command=self.random_operation).grid(column=3, row=2, sticky=W)

        Label(frame_left, text='Request Country').grid(column=0, row=3, sticky=W)
        self.request_country_var = StringVar(frame_left, value=list(countries.keys())[0])
        self.request_country_dropdown = OptionMenu(frame_left, self.request_country_var, *tuple(countries.keys()))
        self.request_country_dropdown.grid(row=3, column=1, sticky=EW)
        Button(frame_left, text='Random', command=self.random_request_country).grid(column=3, row=3, sticky=W)

        Label(frame_left, text='Resource Zone').grid(column=0, row=4, sticky=W)
        self.resource_zone_var = StringVar(frame_left, value=list(zones.keys())[0])
        self.resource_zone_dropdown = OptionMenu(frame_left, self.resource_zone_var, *tuple(zones.keys()))
        self.resource_zone_dropdown.grid(row=4, column=1, sticky=EW)
        Button(frame_left, text='Random', command=self.random_resource_zone).grid(column=3, row=4, sticky=W)

        frame_time = Frame(frame_left)
        frame_time.grid(columnspan=2, row=5, sticky=W)

        Label(frame_time, text='Hour').grid(column=0, row=0)
        self.hour_var = IntVar(frame_time)
        self.hour_dropdown = OptionMenu(frame_time, self.hour_var, *tuple(range(24)))
        self.hour_dropdown.config(width=3)
        self.hour_dropdown.grid(column=1, row=0)

        Label(frame_time, text='Minute').grid(column=2, row=0)
        self.minute_var = IntVar(frame_time)
        self.minute_dropdown = OptionMenu(frame_time, self.minute_var, *tuple(range(60)))
        self.minute_dropdown.config(width=3)
        self.minute_dropdown.grid(column=3, row=0)

        Label(frame_time, text='Second').grid(column=4, row=0)
        self.second_var = IntVar(frame_time)
        self.second_dropdown = OptionMenu(frame_time, self.second_var, *tuple(range(60)))
        self.second_dropdown.config(width=3)
        self.second_dropdown.grid(column=5, row=0)
        Button(frame_left, text='Random', command=self.random_time).grid(column=3, row=5)

        predict_buttons_frame = Frame(frame_left)
        predict_buttons_frame.grid(columnspan=4, row=6, sticky=W)
        Button(predict_buttons_frame, text='Analyse', command=self.predict,
               width=15, height=2, fg='Blue').grid(column=0, row=0)
        Button(predict_buttons_frame, text='Random & Analyse', command=self.random_predict, wraplength=100,
               width=15, height=2, fg='Blue').grid(column=1, row=0)
        Button(predict_buttons_frame, text='Random & Analyse until Anomalous',
               command=self.random_predict_anomalous, wraplength=120,
               width=20, height=2, fg='Blue').grid(column=2, row=0)

        self.data_input_tensor = tf.get_default_graph().get_tensor_by_name('data_input:0')
        self.output_class_tensor = tf.get_default_graph().get_tensor_by_name('test_svm/output:0')
        self.gradient_tensor = tf.get_default_graph().get_tensor_by_name('gradient:0')
        self.histogram_data = np.load('processed_features.npy')
        self.xn = np.linspace(-1, 1, 201)

        self.output = 0
        self.margin = 0
        self.gradient = None

        self.log = Text(frame_left, width=50, height=40)
        scrollbar = Scrollbar(frame_left, command=self.log.yview)
        self.log.config(yscrollcommand=scrollbar.set)
        self.log.insert(END, 'Result')
        self.log.config(state=DISABLED)
        self.log.grid(columnspan=3, row=7)
        scrollbar.grid(column=3, row=7, sticky=NS)

        self.lb = Listbox(frame_middle, width=30, height=30)
        self.lb.insert(END, *tuple(['Feature ' + str(i) for i in range(1, self.gradient_tensor.shape[2] + 1)]))
        self.lb.bind('<<ListboxSelect>>', self.onselect)
        self.lb.grid(row=0, sticky=N)
        self.lb_index = 0

        self.f = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.f.add_subplot(111)
        self.ax.invert_yaxis()
        self.cv = FigureCanvasTkAgg(self.f, frame_right)
        self.cv.show()
        self.cv.get_tk_widget().grid(row=0, column=0, sticky=NW)

        self.f2 = Figure(figsize=(5, 4), dpi=100)
        self.ax2 = self.f2.add_subplot(111)
        self.cv2 = FigureCanvasTkAgg(self.f2, frame_right)
        self.cv2.show()
        self.cv2.get_tk_widget().grid(row=1, column=0, sticky=NW)

    def onselect(self, event):
        try:
            self.lb_index = self.lb.curselection()[0]
            self.plot_1()
            self.plot_2()
        except IndexError:
            pass

    def random_resource(self):
        self.resource_var.set(random.choice(list(self.resources.keys())))

    def random_request_ip(self):
        self.request_ip_var.set('IP Address ' + str(np.random.randint(0, self.request_ip_max + 1)))

    def random_operation(self):
        self.operation_var.set(random.choice(list(self.operations.keys())))

    def random_request_country(self):
        self.request_country_var.set(random.choice(list(countries.keys())))

    def random_resource_zone(self):
        self.resource_zone_var.set(random.choice(list(zones.keys())))

    def random_time(self):
        self.hour_var.set(np.random.randint(0, 24))
        self.minute_var.set(np.random.randint(0, 60))
        self.second_var.set(np.random.randint(0, 60))

    def random_all(self):
        self.random_resource()
        self.random_request_ip()
        self.random_operation()
        self.random_request_country()
        self.random_resource_zone()
        self.random_time()

    def model_predict(self):
        self.data = list()
        self.data.append(self.operations[self.operation_var.get()])
        self.data.append(self.resources[self.resource_var.get()])
        self.data.append(int(self.request_ip_var.get().split(' ')[2]))
        self.data.extend(countries[self.request_country_var.get()])
        self.data.extend(zones[self.resource_zone_var.get()])
        t = 3600*self.hour_var.get() + 60*self.minute_var.get() + self.second_var.get()
        self.data.append(np.cos(2 * np.pi * t/86400))
        self.data.append(np.sin(2 * np.pi * t / 86400))

        self.output = self.tf_session.run(self.output_class_tensor,
                                          feed_dict={self.data_input_tensor: [self.data]})[0][0]
        self.gradient = self.tf_session.run(self.gradient_tensor, feed_dict={self.data_input_tensor: [self.data]})[0]

    def plot_1(self):
        try:
            self.ax.clear()
            ind = np.arange(int(self.gradient_tensor.shape[2]))
            width = 0.8
            self.ax.barh(ind + 1.25 * width, self.gradient.T, width, edgecolor='k')
            self.ax.barh(self.lb_index + 1.25 * width, [self.gradient.T[self.lb_index]], width, color='r')
            self.ax.set_yticks(range(1, int(self.gradient_tensor.shape[2]) + 1))
            self.ax.axvline(0)
            self.ax.set_ylabel('Feature')
            self.ax.set_xlabel('Gradient Score')
            self.cv.draw()

        except AttributeError:
            pass

    def plot_2(self):
        try:
            self.ax2.clear()
            self.ax2.hist(self.histogram_data[:, self.lb_index], edgecolor='k', bins=100, align='left')
            if self.data:
                tensor_x = tf.get_default_graph().get_tensor_by_name('embed/processed_features:0')
                point_x = self.tf_session.run(tensor_x, feed_dict={self.data_input_tensor: [self.data]})
                self.ax2.axvline(point_x[0][self.lb_index], color='r')
            self.cv2.draw()

        except AttributeError:
            pass

    def update_log(self):
        self.log.config(state=NORMAL)
        self.log.delete(1.0, END)
        self.log.insert(INSERT, self.interpret_result(self.output, self.gradient, self.margin))
        self.log.config(state=DISABLED)

    def predict(self):
        self.model_predict()
        self.plot_1()
        self.plot_2()
        self.update_log()

    def random_predict(self):
        self.random_all()
        self.model_predict()
        self.plot_1()
        self.plot_2()
        self.update_log()

    def random_predict_anomalous(self):
        output = 1.0
        while output == 1.0:
            self.random_all()
            self.model_predict()
            output = self.output

        self.plot_1()
        self.plot_2()
        self.update_log()

    def interpret_result(self, output, gradient, margin):
        response = {
            'Input': {
                'Operation': self.operation_var.get(),
                'Resource': self.resource_var.get(),
                'Resource Zone': self.resource_zone_var.get(),
                'Request IP': self.request_ip_var.get(),
                'Request Country': self.request_country_var.get(),
                'Time': ':'.join((str(self.hour_var.get()), str(self.minute_var.get()), str(self.second_var.get())))
            },
            'Output': {
                'Decision': 'Normal' if output == 1.0 else 'Anomalous',
                'Decision function value': round(float(margin), 5),
                'Contribution scores': {}
            }
        }
        ranks = [0] * len(gradient[0])
        for i, x in enumerate(sorted(range(len(gradient[0])), key=lambda y: np.abs(gradient[0][y]))):
            ranks[x] = i + 1
        for i in range(len(gradient[0])):
            if i < 2:
                att = 'Resource'
            elif i < 10:
                att = 'Request IP'
            elif i < 16:
                att = 'Operation'
            elif i < 19:
                att = 'Request Country'
            elif i < 26:
                att = 'Resource Zone'
            else:
                att = 'Time'

            response['Output']['Contribution scores']['Feature ' + str(i + 1)] = {
                'Value': round(float(gradient[0][i]), 5),
                'Rank': len(ranks) + 1 - int(ranks[i]),
                'Attribute': att
            }

        return json.dumps(response, indent=2)


root = Tk()

with tf.Session() as tf_session:
    saved = tf.train.import_meta_graph('model.ckpt.meta')
    saved.restore(tf_session, 'model.ckpt')
    app = App(root, tf_session)
    root.mainloop()
