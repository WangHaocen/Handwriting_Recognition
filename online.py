#!/usr/bin/env python
# coding: utf-8

# In[1]:


mlf_chars = ['ex', 'qu', 'ga', 'do', 'am', 'ti', 'sl', 'lb', 'rb', 'ls', 'rs', 'sr', 'cm', 'mi',
                 'pl', 'pt', 'sp', 'cl', 'sc', 'qm', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                 'n8', 'n9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                 'y', 'z']

CHARS = ['!', '"', '', '', '&', "'", '/', '(', ')', '[', ']', '*', ',', '-',
             '+', '.', ' ', ':', ';', '?', '0', '1', '2', '3', '4', '5', '6', '7',
             '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
             'y', 'z', '%']

blank_index = len(mlf_chars)
non_alphabets = ['!', '"', '', '', '&', "'", '/', '(', ')', '[', ']', '*', ',', '-',
                  '+', '.', ' ', ':', ';', '?', '0', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9', ]

CHARS_SIZE = len(mlf_chars) + 1

blacklists=["h02-037-02", "a02-062-01","a02-062-02","a02-062-03","e05-265z-04",'g04-060-02','j06-278z-04','h04-141z-03']


# In[2]:


class ReadFile():
    def __init__(self,mode=None):
        self.mode = mode
    def get_samples(self):
        if(self.mode=='train'):
            file = open("./data/online/iam-config/trainset.txt")
        elif(self.mode=="valid"):
            file = open("./data/online/iam-config/testset_v.txt")
        elif(self.mode=="eval"):
            file = open("./data/online/iam-config/testset_f.txt")
            
        sample_names = []
        names = ''
        i = 0
        for n in file:
            names =  n.strip(' \n')
            sample_names += [names]
        
        file = open("./data/online/iam-config/t2_labels.mlf")
        samples = []
        current_groundtruth = []
        current_path = ""
        current_sample_name = ""
        
        for n in file:   
            if("scratch" in n):    
                if current_path and current_sample_name in sample_names:
                    current_groundtruth = current_groundtruth[:-1]
                    samples.append(Each_Sample(current_path, current_groundtruth))
                
                current_groundtruth = []
                current_path = ""
                current_sample_name = ""
                scratch = n.strip(' "\n').split('/')
                file_name = scratch[8].split('.')[0]
    
                if(file_name in blacklists):
                    continue
    
                split_name = file_name.split('-')
                file_name = split_name[0] + "/" + split_name[0] + "-" + split_name[1][:3] + "/" + file_name + ".xml"
                current_path = "./data/online/iam/" + file_name
                current_sample_name = split_name[0] + "-" + split_name[1]
                             
            elif("#" in n):
                continue
            else:
                current_groundtruth.append(n.strip('\n'))
        return samples
    


# In[3]:


class Each_Sample(object):
    def __init__(self, xml_path, ground_truth):
        self.xml_path = xml_path
        self.ground_truth = ground_truth
        
    def get_groundtruth_text(self):
        gt =''
        for i in range(len(self.ground_truth)):
            char = self.ground_truth[i]
            index = mlf_chars.index(char)
            gt += CHARS[index]
        return gt
    
    def generate_pointSet(self):
        xml = open(self.xml_path, 'rb').read()
        root = etree.XML(xml)
        wbd, strokeset = root.getchildren()

        sensorlocation = wbd[0].attrib['corner']
        diagonalX = wbd[1].attrib['x']
        diagonalY = wbd[1].attrib['y']
        verticalX = wbd[2].attrib['x']
        verticalY = wbd[2].attrib['y']
        horizontalX = wbd[3].attrib['x']
        horizontalY = wbd[3].attrib['y']

        strokes = []
        stroke_id = 1
        min_time = Decimal(strokeset.getchildren()[0].getchildren()[0].attrib['time'])
        for stroke in strokeset:
            for point in stroke:
                t = (Decimal(point.attrib['time']) - min_time) * 1000
                x = point.attrib['x']
                y = point.attrib['y']
                strokes.append([stroke_id, t, x, y])
            stroke_id += 1
        strokes = np.asarray(strokes, dtype=np.int64)

        right = int(diagonalX)
        bottom = int(diagonalY)  # right, bottom edge
        left = int(verticalX) # left edge
        upper = int(horizontalY)  # upper edge

        strokes[:, 2] = np.subtract(strokes[:, 2], left)
        strokes[:, 3] = np.subtract(strokes[:, 3], upper)
        points = []
        for s in strokes:
            points.append(Point(*s))
        return Each_Sample_PointSet(points=points)
    
    
    def generate_child_pointSet(self):
        df = pd.read_csv(self.xml_path)
        x = df['X'].tolist()
        y = df['Y'].tolist()
     
        dottype = df['DotType'].tolist()
        time = df['Timestamp'].tolist()
        t = []
        t.append(0)
        for i in range(1,len(time)):
            t.append(time[i]-time[i-1])
            
        s = 1
        count = 0
        strokes = []
        for i in range(len(dottype)):
            if(dottype[i]=="PEN_UP"):
                count += 1
                strokes.append(s)
                s +=1
            else:
                strokes.append(s)
        
        for i in range(len(x)):
            x1 = x[i]*100
            y1 = y[i]*100
            x[i] = x1
            y[i] = y1
        
        points = []
        for i in range(len(x)):
            points.append(Point(strokes[i],t[i],x[i],y[i]))
        return Each_Sample_PointSet(points=points,gt=self.get_groundtruth_text())
            
    def __repr__(self):
        return str(self.xml_path.split("/")[-1][:-4])


# In[4]:


import os
from lxml import etree
from decimal import Decimal
import numpy as np
from pylab import rcParams
from matplotlib import pyplot as plt
from rdp import rdp


# In[5]:


class Each_Sample_PointSet:
    def __init__(self, points=None, gt=None):
        self.points = points 
        self.gt = gt
    
    def get_strokegroup(self):
        strokes = []
        for i in range(len(self.points)):
            p = self.points[i]
            strokes.append(p.stroke)
        
        temp_sg = []
        stroke_groups = []

        strokes = set(strokes)
        strokes = list(strokes)
        
        for i in range(len(strokes)):
            n = strokes[i]
            for j in range(len(self.points)):
                p = self.points[j]
                if(p.stroke==n):
                    temp_sg.append(p)
            stroke_groups.append(temp_sg)
            temp_sg = []    
        return stroke_groups
    
    def generate_strokegroup_lines(self):
        lines = []
        strokes = self.get_strokegroup()
        for st in range (len(strokes)):
            s = strokes[st]
            for i in range(len(s) - 1):
                if i == len(s) - 2:
                    lines.append(Line(s[-2], s[-1], eos=True))
                else:
                    lines.append(Line(s[i], s[i + 1]))
        return lines
  
    def preprocessing(self):
        
        y_sd = []
        x_sd = []
        for i in range(len(self.points)):
            p = self.points[i]
            x_sd.append(p.x)
            y_sd.append(p.y)
        
        x_sd = np.array(x_sd)
        y_sd = np.array(y_sd)
        
        x_mean = np.mean(x_sd)
        y_mean = np.mean(y_sd)

        sd_x = np.std(x_sd)
        sd_y = np.std(y_sd)
     
        for p in range (len(self.points)):
            point = self.points[p]
            point.x = (point.x - x_mean)/sd_y
            point.y = (point.y - y_mean)/ sd_y
            self.points[p] = point
            
        x = []
        y = []
        l = []
        for i in range(len(self.points)):
            p = self.points[i]
            new_p = [p.x,p.y]
            l.append(new_p)
       # print(l)
        
        x_new = []
        y_new = []
        new_l = rdp(l,epsilon=0.03)
        for i in range(len(new_l)):
            x_new.append(new_l[i][0])
            y_new.append(new_l[i][1])
            
        new_p = []
        for i in range(len(self.points)):
            x = self.points[i].x
            if x in x_new:
                new_p.append(self.points[i])
            
        self.points = new_p
        self.linear_interpolation(d=0.15)
        
    def preprocessing_offline(self):
        
        y_sd = []
        x_sd = []
        for i in range(len(self.points)):
            p = self.points[i]
            x_sd.append(p.x)
            y_sd.append(p.y)
        
        x_sd = np.array(x_sd)
        y_sd = np.array(y_sd)
        
        x_mean = np.mean(x_sd)
        y_mean = np.mean(y_sd)

        sd_x = np.std(x_sd)
        sd_y = np.std(y_sd)
     
        for p in range (len(self.points)):
            point = self.points[p]
            point.x = (point.x - x_mean)/sd_y
            point.y = (point.y - y_mean)/ sd_y
            self.points[p] = point
            
        x = []
        y = []
        l = []
        for i in range(len(self.points)):
            p = self.points[i]
            new_p = [p.x,p.y]
            l.append(new_p)
       # print(l)
        
        x_new = []
        y_new = []
        new_l = rdp(l,epsilon=0.03)
        for i in range(len(new_l)):
            x_new.append(new_l[i][0])
            y_new.append(new_l[i][1])
            
        new_p = []
        for i in range(len(self.points)):
            x = self.points[i].x
            if x in x_new:
                new_p.append(self.points[i])
            
        self.points = new_p
        self.linear_interpolation(d=0.05)
        
        rcParams['figure.figsize'] = 2, 2
        groups = self.get_strokegroup()
        x = []
        y = []
       
        for i in range(len(groups)):
            g = groups[i]
            for j in range(len(g)):
                x.append(g[j].x)
                y.append(-g[j].y)
                
            plt.plot(x, y, '.', linewidth=2, color=(0, 0, 0))
            x=[]
            y=[]
       
        plt.gca().set_aspect(aspect=0.9)
        plt.axis('off')
        plt.savefig("application.png", bbox_inches='tight',pad_inches = 0)
        plt.show()
        plt.close()
        
   
    def linear_interpolation(self, d):
        strokes = self.get_strokegroup()
        interp_points = []
        for s in range(len(strokes)):
            stroke = strokes[s]
            if len(stroke) > 0:
                interp_points.append(stroke[0])
            for i in range(1, len(stroke)):
                line = Line(interp_points[-1], stroke[i])
                l = line.length()
                
                if l > d:
                    f = d / l
                    iteration = int(l / d)
                    for j in range(iteration):
                        iterated = f*(j+1)
                        p1 = line.p1
                        p2 = line.p2
                        time_diff = p2.time - p1.time
                        
                        new_vec = line.vec()*iterated
                        new_time_diff = time_diff*iterated
                        new_coord = p1.coordinates() + new_vec
                        new_x = new_coord[0]
                        new_y = new_coord[1]
                        
                        point = Point(p1.stroke,p1.time+new_time_diff,new_x,new_y)
                        interp_points.append(point)

                elif l == d:
                    interp_points.append(stroke[i])
                elif l < d:
                    continue
            interp_points.append(stroke[-1])
        self.points = interp_points
        return interp_points
    
    def generate_features(self):
        self.preprocessing()
        lines = self.generate_strokegroup_lines()
        features = []
        
        for i in range(len(lines)):
            l = lines[i]
            f = l.get_features()
            features.append(f)
        features = np.array(features) 
        
        dim = features.shape[0]
        result = np.zeros((dim, features.shape[1]))
        result[:features.shape[0], :] = features
        return result
    
    def generate_children_features(self):
        self.preprocessing_children()
        lines = self.generate_strokegroup_lines()
        features = []
        
        for i in range(len(lines)):
            l = lines[i]
            f = l.get_features()
            features.append(f)
        features = np.array(features) 
        
        dim = features.shape[0]
        result = np.zeros((dim, features.shape[1]))
        result[:features.shape[0], :] = features
        return result
  
    def plot(self):
        rcParams['figure.figsize'] = 10, 20
        groups = self.get_strokegroup()
        x = []
        y = []
       
        for i in range(len(groups)):
            g = groups[i]
            for j in range(len(g)):
                x.append(g[j].x)
                y.append(-g[j].y)
                
            plt.plot(x, y, '.', linewidth=2, color=(0, 0, 0))
            x=[]
            y=[]
    
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
    def plot_children(self):
        rcParams['figure.figsize'] = 2, 2
        groups = self.get_strokegroup()
        x = []
        y = []
       
        for i in range(len(groups)):
            g = groups[i]
            for j in range(len(g)):
                x.append(g[j].x)
                y.append(-g[j].y)
                
            plt.plot(x, y, '.', linewidth=2, color=(0, 0, 0))
            x=[]
            y=[]
        gt = self.gt
        plt.gca().set_aspect(aspect=0.9)
        plt.axis('off')
        plt.savefig('./data/png/' + gt + ".png", bbox_inches='tight',pad_inches = 0)
        plt.show()
        plt.close()
        
    def __repr__(self):
        return str("PointSet: " + str(len(self.points)) + " points")


# In[6]:


class Point:

    def __init__(self, stroke, time, x, y):
        self.stroke = stroke
        self.time = time
        self.x = x
        self.y = y

    def coordinates(self):
        return np.array([self.x, self.y])

    def __repr__(self):
        return str("Point stroke=" + str(self.stroke) + " time=" + str(self.time) + " x=" + str(self.x) + " y=" + str(self.y))


# In[7]:


class Line:
    def __init__(self, p1, p2, eos=False):
        self.p1 = p1
        self.p2 = p2
        self.eos = eos

    def vec(self):
        return self.p2.coordinates() - self.p1.coordinates()

    def length(self):
        return np.linalg.norm(self.vec())

    def get_features(self):
        x_start = self.p1.x
        y_start = self.p1.y
        delta_x, delta_y = self.vec()
        if(self.eos==False):
            down = 1
            up = 0
        elif(self.eos==True):
            down = 0
            up = 1
        
        time_diff = self.p2.time - self.p1.time
        return np.array([x_start, y_start, delta_x, delta_y, down, up])

    def __repr__(self):
        return str("Line" + self.p1.__repr__() + '\n' + self.p2.__repr__() + '\n>')



def pad_2d(x, pad_to, pad_value):
    result = np.ones((pad_to, x.shape[1])) * pad_value
    result[:x.shape[0], :] = x
    return result

def pad_1d(x, pad_to, pad_value):
    result = np.ones(pad_to) * pad_value
    result[:x.shape[0]] = x
    return result


# In[19]:


from tensorflow.keras.utils import Sequence
from random import randint


# In[20]:


class Array_to_Sequence(Sequence):
    def __init__(self, batch_size=1, pad_to=None, inout_ratio=4, mode=None):
        self.mode = mode
        if(mode=="train"):
            reader = ReadFile("train")
            self.npz_dir = "npz-train"
        elif(mode=="valid"):
            reader = ReadFile("valid")
            self.npz_dir = "npz-valid"
        elif(mode=="eval"):
            reader = ReadFile("eval")
            self.npz_dir = "npz-eval"
        print(self.npz_dir)
        self.samples = np.asarray(reader.get_samples())
        print(len(self.samples))
        
        # xs: features
        # ys: ground truths
        self.xs = []
        self.ys = []
        self.adaptive_pad = True

        # Load features from npz or preprocess from scratch
        
        for s in self.samples:
            f_split = s.xml_path.split('/')
            f_split[-4] = self.npz_dir
            f_split[-1] = f_split[-1][:-3] + 'npz'
            f = '/'.join(f_split)
            data = np.load(f)
            self.xs.append(data['x'])
            self.ys.append(data['y'])
        self.xs = np.asarray(self.xs)
        self.ys = np.asarray(self.ys)

        self.n = len(self.samples)
        # Indices for shuffling
        self.indices = np.arange(self.n)
        np.random.shuffle(self.indices)
        self.batch_size = batch_size
        # If pred, generate only xs

        # Manually define pad value
        if pad_to:
            self.x_pad, self.y_pad = pad_to
            self.adaptive_pad = False

        # Else pad to match the longest sample in batch
        else:
            self.adaptive_pad = True
        # How much the TDNN scale down the input
        self.inout_ratio = inout_ratio
        
        random_index = randint(0, 1000)
        a1 = self.xs[random_index]
        a2 = reader.get_samples()[random_index].generate_pointSet().generate_features()
        print("check random npz and features: ", np.array_equal(a1,a2))

        
        
    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    # Get a batch
    def __getitem__(self, idx):
        # batch indices
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sample = self.samples[inds]
        batch_xs = self.xs[inds]
        batch_ys = self.ys[inds]

        # pad depending on the longest sample of each batch
        if self.adaptive_pad:
            max_len_x = max([len(i) for i in batch_xs])
            y_pad = int(np.ceil(max_len_x / self.inout_ratio))
            x_pad = y_pad * self.inout_ratio

        # Pad with given pad_x and pad_y value
        else:
            x_pad = self.x_pad
            y_pad = self.y_pad

        # features
        inputs = np.array([pad_2d(x, pad_to=x_pad, pad_value=0)
                           for x in batch_xs])
        
        mlf_label = []
        for i in range(len(batch_ys)):
            ys = batch_ys[i]
            l = []
            for j in range(len(ys)):
                m = mlf_chars.index(ys[j])
                l.append(m)
            l = np.array(l)
            mlf_label.append(l)
        mlf_label = np.array(mlf_label)
        
        labels = []
        
        for i in range(len(mlf_label)):
            y = mlf_label[i]
            labels.append(pad_1d(y, pad_to=y_pad, pad_value=-1))
        labels = np.array(labels)
      
        # Length of network output
        ypred_length = np.array([y_pad
                                 for _ in batch_sample])[:, np.newaxis]
        # Number of chars in ground truth
        ytrue_length = np.array([len(s.ground_truth)
                                 for s in batch_sample])[:, np.newaxis]

        # Training/evaluation sequence
        return {'xs': inputs,
                'ys': labels,
                'ypred_length': ypred_length,
                'ytrue_length': ytrue_length}, labels

    def sample_at_idx(self, idx, pad=10):
        idx = self.indices[idx]
        return self.sample_at_absolute_idx(idx, pad=pad)

    # regardless of shuffling    
    def sample_at_absolute_idx(self, idx, pad=10):
        pointset = self.samples[idx].generate_pointSet()
        gt = self.samples[idx].ground_truth
        ground_truth = ''
        for i in range(len(gt)):
            char = gt[i]
            index = mlf_chars.index(char)
            ground_truth += CHARS[index]
        
        feature = self.xs[idx]
        
        x = self.xs[idx]
        shape = feature.shape[0]+pad
        
        result = np.ones((shape, x.shape[1])) * 0
        result[:x.shape[0], :] = x
        feature = result
        feature = np.asarray([feature])
        return feature, ground_truth, pointset


    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    # Get xs and ys which match the current permutation defined by indices
    def get_xy(self):
        xs = []
        for idx in self.indices:
            arr = np.array(self.xs[idx])
            xs.append(arr)
        
        ys = []
        for idx in self.indices:
            gt = self.ys[idx]
            ground_truth = ''
            for i in range(len(gt)):
                char = gt[i]
                index = mlf_chars.index(char)
                ground_truth += CHARS[index]
            ys.append(ground_truth)
        ys = np.array(ys)
        return xs, ys

    def gen_iter(self):
        for i in range(len(self)):
            yield self[i]


class BeamSearch():
    def __init__(self,width):
        self.width = width
    
    def decode(self,rnn_out,top_n=1):
        
        epsilon = 0.0000007
        samples = rnn_out.shape[0]
        length = np.ones(samples)*rnn_out.shape[1]
        input_length = math_ops.to_int32(length)
        rnn_out = math_ops.log(array_ops.transpose(rnn_out, perm=[1, 0, 2]) + epsilon)
        
        decoded, log_prob = ctc.ctc_beam_search_decoder(inputs=rnn_out,
                                                        sequence_length=input_length,
                                                        beam_width=self.width,
                                                        top_paths=top_n,
                                                        merge_repeated=False)
        
        decoded_index = []
        for i in range(len(decoded)):
            decode = decoded[i]
            dense = sparse_ops.sparse_to_dense(decode.indices,decode.dense_shape,decode.values,default_value=-1)
            decoded_index.append(dense)
        
        candidates = []
        for i in range(len(decoded_index)):
            index = decoded_index[i]
            candi = K.eval(index)
            candidates.append(candi)

        pred = [[] for _ in range(samples)]
        for k in range(samples):
            for c in candidates:
                pred[k].append(c[k])
                
        preds = []
        preds_2 = []
        l = []
        for i in range(len(pred)):
            p = pred[i]
            char = ''
            
            for j in range(len(p)):
                arr = p[j]
                for k in range(len(arr)):
                    if(arr[k]==-1 or arr[k]==len(CHARS)):
                        char +=''
                    else:
                        char += CHARS[arr[k]]
                l.append(char)
                char = ''
            preds_2.append(l)
            
        return preds_2


# In[27]:


from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Input, Dense, Activation,       LSTM,GRU, Lambda, BatchNormalization, Bidirectional
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ctc_decode
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import sparse_ops, math_ops, array_ops
from tensorflow.keras.optimizers import Adam
import editdistance
import Levenshtein  as lv


# In[28]:


class Online_Model(object):
    def __init__(self, chars= CHARS, preload=False, rnn=None, decoder=None):
        self.decoder = decoder
        self.rnn = LSTM
        if decoder is None:
            self.decoder = BeamSearch(25)
        
        self.chars = chars
        self.char_size = len(chars) + 1
        self.model = self.generate_model()
        self.pred_model = self.get_premodel("softmax")
        self.compile()
        
        if preload:
            self.pretrained = "./model/online/online_without_children_NoTimediff_standar_rdg003_resample015_withxy_batch8_final.h5"
            print("preloading model weights from " + self.pretrained)
            self.load_weights(file_name=self.pretrained)
        

    def get_loss(self):
        return {'ctc': lambda y_true, y_pred: y_pred}

    def generate_model(self):
        input_shape = (None, 6)
    
        inputs = Input(shape=input_shape, dtype='float32', name='xs')
        inner = inputs
  
        conv1d_1 = Conv1D(60, 7, padding="same", kernel_initializer='he_normal')(inner)
        batch_1 = BatchNormalization()(conv1d_1)
        relu_1 = Activation('relu')(batch_1)

        conv1d_2 = Conv1D(90, 7, padding="same", kernel_initializer='he_normal')(relu_1)
        batch_2 = BatchNormalization()(conv1d_2)
        relu_2 = Activation('relu')(batch_2)
        
        conv1d_3 = Conv1D(120, 5, padding="same", kernel_initializer='he_normal')(relu_2)
        batch_3 = BatchNormalization()(conv1d_3)
        relu_3 = Activation('relu')(batch_3)
        
        pool_1 = AveragePooling1D(pool_size=2)(relu_3)

        conv1d_4 = Conv1D(120, 3, padding="same", kernel_initializer='he_normal')(pool_1)
        batch_4 = BatchNormalization()(conv1d_4)
        relu_4 = Activation('relu')(batch_4)
        
        conv1d_5 = Conv1D(160, 3, padding="same", kernel_initializer='he_normal')(relu_4)
        batch_5 = BatchNormalization()(conv1d_5)
        relu_5 = Activation('relu')(batch_5)
        
       
        conv1d_6 = Conv1D(200, 3, padding="same", kernel_initializer='he_normal')(relu_5)
        batch_6 = BatchNormalization()(conv1d_6)
        relu_6 = Activation('relu')(batch_6)
        
        pool_2 = AveragePooling1D(pool_size=2)(relu_6)
   
        bidlstm_1 = Bidirectional(LSTM(64, return_sequences=True, dropout = 0.2))(pool_2)
        bidlstm_2 = Bidirectional(LSTM(64, return_sequences=True, dropout = 0.2))(bidlstm_1)
        bidlstm_3 = Bidirectional(LSTM(64, return_sequences=True, dropout = 0.2))(bidlstm_2)
      
        dense = Dense(CHARS_SIZE+1, kernel_initializer='he_normal')(bidlstm_3)
        y_pred = Activation('softmax', name='softmax')(dense)

        labels = Input(name='ys',shape=[None], dtype='float32')
        input_length = Input(name='ypred_length', shape=[1], dtype='int64')
        label_length = Input(name='ytrue_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,),
                          name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[inputs, labels, input_length, label_length],
                      outputs=loss_out)
        return model

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def get_premodel(self, layer_name):
        pre_model = Model(inputs=self.model.get_layer("xs").output,
                         outputs=self.model.get_layer(layer_name).output)
       
        optimizer = Adam(learning_rate=0.0001)
        pre_model.compile(loss={layer_name: lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        return pre_model

    def fit(self, train_seq, test_seq, epochs=100, earlystop=10):
        
        filepath="online_without_children_NoTimediff_standar_rdg003_resample015_withxy_batch8.h5"
        early = tf.keras.callbacks.EarlyStopping(patience=earlystop)

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.history = self.model.fit(
            train_seq,
            validation_data=test_seq,
            shuffle=True,
            verbose=1,
            epochs=epochs,
            callbacks=[checkpoint, early]
        )
    
    def get_history(self):
        return self.history

    def predict_softmax(self, x):
        if isinstance(x, Sequence) and x.batch_size == 1:
            print("predicting softmax for sequence with batch size: 1, will return list of ndarray.")
            sm = []
            gen = x.gen_iter()
            for b in tqdm(gen, total=len(x)):
                sm.append(self.pred_model.predict(b, verbose=0)[0])
        elif isinstance(x, Sequence):
            sm = self.pred_model.predict_generator(x, verbose=1)
        else:
            sm = self.pred_model.predict(x, verbose=1)
        
        return sm

    def predict(self, x, decoder=None, top=1):
        if decoder is None:
            decoder = self.decoder

        softmaxs = self.predict_softmax(x)
        pred = decoder.decode(rnn_out=softmaxs, top_n=top)
        if top == 1:
            try:
                pred = [p[0] for p in pred]
            except IndexError:
                print("Index Error: {}".format(pred))
        return pred


    def compile(self):
        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(loss=self.get_loss(), optimizer=optimizer)

    def save_weights(self, file_name=None):
        self.model.save_weights(file_name)

    def load_weights(self, file_name=None):
        self.model.load_weights(file_name)
        self.compile()

    def get_model_summary(self):
        return self.model.summary()
    
    def character_error_rate(self,y_true, y_pred):
        cer = 0
        for i in range(len(y_true)):
            leven = lv.distance(y_true[i], y_pred[i])
            char = len(y_true[i])
            cer += leven / char
        CER = cer / len(y_true)
        return CER
    
    def word_error_rate(self,y_true, y_pred):
        total_wer = 0
        for i in range(len(y_true)):
            gt = y_true[i]
            pred = y_pred[i]
            gt = gt.split(" ")
            pred = pred.split(" ")
            words = list(set(gt + pred))
            
            index_gt = []
            for w in gt:
                index_gt.append(words.index(w))
            index_pred = []
            for w in pred:
                index_pred.append(words.index(w))
                
            leven = lv.distance(index_gt, index_pred)
            wer = leven / len(index_gt)
            total_wer += wer
        avg_wer = total_wer / len(y_true)
        return avg_wer





