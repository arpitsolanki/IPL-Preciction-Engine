import streamlit as st
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
  st.title("IPL Prediction Engine ")
  st.markdown("This application uses detailed match by match data for IPL matches available at cricsheet.org to train a ML model based on matches going hack to 2008 & predicts outcome of upcoming season. The model is re-run after every single match in the tournament to update win probabilities")
  data=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/final_output.csv')
  gw_list=data['date'].unique().tolist()
  menu=gw_list
  choice = st.sidebar.selectbox('Game Date',menu)  
  data_fil=data.loc[data.date==choice]
  data_fil=data_fil.reset_index(drop=True)
#  st.write(data_fil)

  for i in range(data_fil.shape[0]):
    st.subheader('MATCH OF THE DAY')

    team_x=data_fil.loc[i,'team_name_x']
    team_y=data_fil.loc[i,'team_name_y']

    team_x=team_x.replace(" ",'%20')
    team_y=team_y.replace(" ",'%20')
    
    pred_x=data_fil.loc[i,'pred_team_x']
    pred_y=data_fil.loc[i,'pred_team_y']
    
    col= st.beta_columns(2)

    img_path='https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/Logos/'+team_x+'.jpg'
    img=io.imread(img_path)  
    img_path='https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/Logos/'+team_y+'.jpg'
    img1=io.imread(img_path)  
    
    col[0].image(img,width=250)
    col[1].image(img1,width=250)
    
    var="VENUE"
    st.markdown("<h3 style='text-align: center; color: black;'>"+var+"</h3>", unsafe_allow_html=True)

    var=data_fil.loc[i,'venue']
    st.markdown("<h4 style='text-align: center; color: black;'>"+var+"</h4>", unsafe_allow_html=True)

    var="WIN PROBABILITY"
    st.markdown("<h3 style='text-align: center; color: black;'>"+var+"</h3>", unsafe_allow_html=True)

  #  st.subheader('WIN PROBABILITY')
    col= st.beta_columns((1,2,1,2))
    col[1].header("{0:.0%}".format(pred_x))
    col[3].header("{0:.0%}".format(pred_y))

    var="WINNING TEAM"
    st.markdown("<h3 style='text-align: center; color: black;'>"+var+"</h3>", unsafe_allow_html=True)

    var=data_fil.loc[i,'winning_team']
    st.markdown("<h4 style='text-align: center; color: black;'>"+var+"</h4>", unsafe_allow_html=True)

    
if __name__ == '__main__':
	main()
