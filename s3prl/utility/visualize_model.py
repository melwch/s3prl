def display_num_params(net):
  num_param = 0
  
  for param in net.parameters():
    num_param += param.numel()
  print(f"There are {num_param:,} ({num_param / 1e6:.2f} million) parameters in this neural network")