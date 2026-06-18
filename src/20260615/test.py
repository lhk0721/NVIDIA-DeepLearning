from tensorflow.keras.models import load_model
best_cnn_model = load_model('models/catdof_bestmodel.keras')

best_cnn_model.summary()