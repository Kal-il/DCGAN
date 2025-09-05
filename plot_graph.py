import pandas as pd
import plotly.express as px
import io
import os

# directory = "experiments/test_1000_epochs_4x4_base_64x64_gen_lr_adjusted"
directory = "experiments/test_2_1000_epochs_4x4_base_64x64_gen_lr_adjusted_to_0_00015"
path_to_csv = os.path.join(directory, "training_log.csv")

df = pd.read_csv(path_to_csv)
total_time_sec = df['time_sec'].sum()
total_time_min = total_time_sec / 60
total_time_hour = total_time_min / 60

# Reformatar o DataFrame para facilitar a plotagem com Plotly
# Usamos a função 'melt' para transformar as colunas 'gen_loss' e 'disc_loss' em linhas
df_melted = df.melt(id_vars=['epoch'], value_vars=['gen_loss', 'disc_loss'],
                    var_name='Tipo de Perda', value_name='Valor da Perda')

# Renomear os valores para uma legenda mais clara
df_melted['Tipo de Perda'] = df_melted['Tipo de Perda'].map({
    'gen_loss': 'Perda do Gerador (gen_loss)',
    'disc_loss': 'Perda do Discriminador (disc_loss)'
})

# Criar o gráfico de linha interativo
fig = px.line(df_melted,
              x='epoch',
              y='Valor da Perda',
              color='Tipo de Perda',
              title=f'Análise de Perda do Modelo<br><sup>Tempo Total de Treinamento: {total_time_hour:.2f} horas, {total_time_min:.2f} minutos e {total_time_sec:.2f} segundos</sup>',
              labels={
                  "epoch": "Época",
                  "Valor da Perda": "Valor da Perda (Loss)",
                  "Tipo de Perda": "Legenda"
              },
              template='plotly_white', # Usando um template limpo para artigos
              markers=False) # Adiciona marcadores nos pontos de dados

# Customizar a aparência do gráfico para uma publicação
fig.update_layout(
    font_family="Times New Roman",
    title_font_size=22,
    xaxis=dict(
        title_font_size=18,
        tickfont_size=14,
    ),
    yaxis=dict(
        title_font_size=18,
        tickfont_size=14,
    ),
    legend=dict(
        font_size=14,
        title_font_size=16
    )
)

# Para visualizar o gráfico em um ambiente como Jupyter Notebook, use:
# fig.show()

# Para salvar em um arquivo HTML interativo:
fig.write_html("grafico_perdatest_lr_adjusted_to_0_00015.html")

# Para salvar em uma imagem estática (PNG, JPG, PDF), você pode precisar
# instalar um pacote adicional: pip install -U kaleido
# fig.write_image("grafico_perda.png")
