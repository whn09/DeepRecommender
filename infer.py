# Copyright (c) 2017 NVIDIA Corporation
import torch
import argparse
import copy
from reco_encoder.data import input_layer
from reco_encoder.model import model
from torch.autograd import Variable
from pathlib import Path
from math import sqrt
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import svm, datasets  
from sklearn.metrics import roc_curve, auc  ###计算roc和auc  
from sklearn import cross_validation  

parser = argparse.ArgumentParser(description='RecoEncoder')

parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                    help='dropout drop probability')
parser.add_argument('--constrained', action='store_true',
                    help='constrained autoencoder')
parser.add_argument('--skip_last_layer_nl', action='store_true',
                    help='if present, decoder\'s last layer will not apply non-linearity function')
parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                    help='hidden layer sizes, comma-separated')
parser.add_argument('--path_to_train_data', type=str, default="", metavar='N',
                    help='Path to training data')
parser.add_argument('--path_to_eval_data', type=str, default="", metavar='N',
                    help='Path to evaluation data')
parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                    help='type of the non-linearity used in activations')
parser.add_argument('--save_path', type=str, default="autorec.pt", metavar='N',
                    help='where to save model')
parser.add_argument('--predictions_path', type=str, default="out.txt", metavar='N',
                    help='where to save predictions')

args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available() # global flag
if use_gpu:
    print('GPU is available.') 
else: 
    print('GPU is not available.')

def do_eval(encoder, evaluation_data_layer):
  encoder.eval()
  denom = 0.0
  total_epoch_loss = 0.0
  for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
    inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
    targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
    outputs = encoder(inputs)
    loss, num_ratings = model.MSEloss(outputs, targets)
    total_epoch_loss += loss.data[0]
    denom += num_ratings.data[0]
  return sqrt(total_epoch_loss / denom)
    
def main():
  params = dict()
  params['batch_size'] = 1
  params['data_dir'] =  args.path_to_train_data
  params['major'] = 'users'
  params['itemIdInd'] = 1
  params['userIdInd'] = 0
  print("Loading training data")
  data_layer = input_layer.UserItemRecDataProvider(params=params)
  print("Data loaded")
  print("Total items found: {}".format(len(data_layer.data.keys())))
  print("Vector dim: {}".format(data_layer.vector_dim))

  print("Loading eval data")
  eval_params = copy.deepcopy(params)
  # must set eval batch size to 1 to make sure no examples are missed
  eval_params['batch_size'] = 1
  eval_params['data_dir'] = args.path_to_eval_data
  eval_data_layer = input_layer.UserItemRecDataProvider(params=eval_params,
                                                        user_id_map=data_layer.userIdMap,
                                                        item_id_map=data_layer.itemIdMap)

  rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] + [int(l) for l in args.hidden_layers.split(',')],
                               nl_type=args.non_linearity_type,
                               is_constrained=args.constrained,
                               dp_drop_prob=args.drop_prob,
                               last_layer_activations=not args.skip_last_layer_nl)

  path_to_model = Path(args.save_path)
  if path_to_model.is_file():
    print("Loading model from: {}".format(path_to_model))
    rencoder.load_state_dict(torch.load(args.save_path))

  print('######################################################')
  print('######################################################')
  print('############# AutoEncoder Model: #####################')
  print(rencoder)
  print('######################################################')
  print('######################################################')
  rencoder.eval()
  if use_gpu: rencoder = rencoder.cuda()
  
  inv_userIdMap = {v: k for k, v in data_layer.userIdMap.items()}
  inv_itemIdMap = {v: k for k, v in data_layer.itemIdMap.items()}

  eval_data_layer.src_data = data_layer.data
  y_test = []
  y_score = []
  with open(args.predictions_path, 'w') as outf:
    for i, ((out, src), majorInd) in enumerate(eval_data_layer.iterate_one_epoch_eval(for_inf=True)):
      inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
      targets_np = out.to_dense().numpy()[0, :]
      outputs = rencoder(inputs).cpu().data.numpy()[0, :]
      non_zeros = targets_np.nonzero()[0].tolist()
      major_key = inv_userIdMap [majorInd]
      for ind in non_zeros:
        outf.write("{}\t{}\t{}\t{}\n".format(major_key, inv_itemIdMap[ind], outputs[ind], targets_np[ind]))
        y_test.append(targets_np[ind]-1)
        y_score.append(outputs[ind]-1)
      if i % 10000 == 0:
        print("Done: {}".format(i))
        
  eval_loss = do_eval(rencoder, eval_data_layer)
  print('EVALUATION LOSS: {}'.format(eval_loss))
    
  try:
      fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率  
      roc_auc = auc(fpr,tpr) ###计算auc的值  
      print('AUC:', roc_auc)
      plt.figure()  
      lw = 2  
      plt.figure(figsize=(10,10))  
      plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
      plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
      plt.xlim([0.0, 1.0])  
      plt.ylim([0.0, 1.05])  
      plt.xlabel('False Positive Rate')  
      plt.ylabel('True Positive Rate')  
      plt.title('ROC')  
      plt.legend(loc="lower right")
      plt.savefig(args.predictions_path+'.png')
      #plt.show() 
  except:
      pass

if __name__ == '__main__':
  main()


