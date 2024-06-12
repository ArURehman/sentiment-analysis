import tensorflow as tf

def BertSentimentAnalysis(bert_model, num_classes):
    model = tf.keras.Sequential()
    model.add(bert_model)
    model.add(tf.keras.layers.Lambda(lambda x: x[1]))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model