import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from service import model_runner

def main() -> None:
    
    DISABLED = False    
    st.title('Sentiment Analysis')    
    user_input = st.text_area('', max_chars=250, placeholder='Enter text here', disabled=DISABLED)
    if st.button('Analyze', disabled=DISABLED):
        DISABLED = True
        if user_input:
            with st.spinner('Analyzing...'):
                sentiment, prediction = model_runner.run(user_input)
                st.text(f'Sentiment: {sentiment}')
                st.bar_chart(np.array(prediction[0]), height=400, width=400, use_container_width=True)
                st.dataframe(pd.DataFrame(prediction, columns=model_runner.SENTIMENTS))
                fig, ax = plt.subplots()
                ax.pie(np.array(prediction[0]), labels=model_runner.SENTIMENTS, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig, clear_figure=True)
        else:
            st.error('Please enter some text')
        DISABLED = False

if __name__ == '__main__':
    main()