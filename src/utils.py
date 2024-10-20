def BIO_to_inner(label):
    if(label[0]=='B'): label = 'I' + label[1:]
    return label
