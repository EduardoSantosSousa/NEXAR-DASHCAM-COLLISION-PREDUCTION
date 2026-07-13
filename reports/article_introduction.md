# 1. Introdução

Os acidentes de trânsito permanecem entre os principais desafios relacionados à segurança e à mobilidade em escala global. Segundo a Organização Mundial da Saúde, aproximadamente 1,19 milhão de pessoas morrem anualmente em decorrência de acidentes viários, enquanto entre 20 e 50 milhões sofrem lesões não fatais, muitas delas associadas a incapacidades permanentes (WORLD HEALTH ORGANIZATION, 2023). Esses eventos também produzem consequências econômicas e sociais significativas, relacionadas aos custos hospitalares, aos danos materiais e à perda de produtividade.

Nesse contexto, o desenvolvimento de sistemas avançados de assistência ao condutor — *Advanced Driver Assistance Systems* (ADAS) — e de tecnologias voltadas aos veículos autônomos tem ampliado o interesse por métodos capazes de identificar situações perigosas antes que um acidente aconteça. Entre as diferentes fontes de informação disponíveis, destacam-se os vídeos obtidos por câmeras veiculares, conhecidas como *dashcams*, que registram continuamente a perspectiva frontal do veículo e fornecem informações sobre a via, os demais usuários, as condições ambientais e a dinâmica dos objetos presentes na cena.

Quando combinados com técnicas de visão computacional e aprendizado de máquina, esses vídeos podem ser utilizados não apenas para reconhecer acidentes já ocorridos, mas também para estimar antecipadamente a probabilidade de uma colisão. Essa tarefa é denominada na literatura como antecipação de acidentes de trânsito, ou *Traffic Accident Anticipation*, e busca identificar indícios de perigo antes da ocorrência efetiva do evento.

A antecipação de acidentes apresenta desafios adicionais em comparação com a classificação convencional de vídeos. O modelo deve reconhecer indícios visuais ainda sutis e incompletos, como mudanças inesperadas de trajetória, aproximações perigosas, frenagens, oclusões, comportamento de pedestres e interações entre múltiplos veículos. Além disso, existe um compromisso entre a antecedência e a confiabilidade da previsão: quanto mais cedo o alerta é emitido, menor é a quantidade de evidências visuais disponíveis, o que pode aumentar a ocorrência de falsos positivos.

Chan et al. (2016) apresentaram uma das abordagens pioneiras para antecipação de acidentes em vídeos de *dashcam*, utilizando uma rede neural recorrente combinada com um mecanismo de atenção espacial dinâmica. A proposta buscava identificar objetos e regiões relevantes da cena e, simultaneamente, modelar a evolução temporal das situações de risco. O estudo demonstrou que a análise conjunta das informações espaciais e temporais é importante para reconhecer indícios de acidentes antes de sua ocorrência.

Posteriormente, Bao, Yu e Kong (2020) propuseram uma abordagem baseada em aprendizagem relacional espaço-temporal e estimativa de incerteza. O modelo utiliza redes convolucionais em grafos e redes recorrentes para representar as relações entre os agentes presentes na cena, além de incorporar redes neurais bayesianas para reduzir previsões excessivamente confiantes em situações ambíguas. Esse tipo de abordagem evidencia que, além da aparência visual dos objetos, as relações espaciais e temporais entre veículos, pedestres e outros elementos da cena são fundamentais para a antecipação de acidentes.

Apesar desses avanços, a previsão antecipada de colisões continua sendo um problema complexo. Modelos baseados em frames isolados podem não representar adequadamente a dinâmica da cena, enquanto arquiteturas temporais mais sofisticadas geralmente exigem maior quantidade de dados, maior capacidade computacional e processos de treinamento mais complexos. Também existem dificuldades relacionadas à variabilidade das condições de tráfego, às diferenças entre ambientes urbanos e rodoviários, às mudanças de iluminação, às condições climáticas e às oclusões presentes nos vídeos.

Outro aspecto importante está relacionado à avaliação dos modelos. Em aplicações de segurança veicular, avaliar somente a classificação final entre ocorrência e não ocorrência de colisão não é suficiente. Uma previsão correta produzida apenas no instante da colisão possui utilidade prática limitada, pois pode não fornecer tempo suficiente para que o condutor ou um sistema automatizado execute uma ação preventiva. Dessa forma, a avaliação deve considerar simultaneamente a capacidade de classificação, a antecedência do alerta, a evolução temporal do risco e a frequência de falsos positivos e falsos negativos.

O conjunto de dados **Nexar Dashcam Collision Prediction** foi desenvolvido para apoiar pesquisas relacionadas à análise de eventos de trânsito, à previsão antecipada de colisões e à segurança de veículos autônomos. O conjunto contém 1.500 vídeos anotados, com aproximadamente 40 segundos de duração cada, abrangendo cenários de condução normal, colisões e quase-colisões. Os vídeos também apresentam diferentes condições ambientais, incluindo variações de iluminação, clima e tipo de ambiente viário (MOURA; ZHU; ZVITIA, 2025).

Para os vídeos que apresentam colisões ou quase-colisões, o conjunto de dados fornece anotações temporais relacionadas ao instante do evento e ao denominado *alert time*. O *alert time* representa o primeiro momento em que um observador atento poderia reconhecer que a situação estava evoluindo para um evento perigoso. Essa anotação permite avaliar não apenas se o modelo identificou corretamente o risco, mas também com que antecedência a previsão foi realizada (MOURA; ZHU; ZVITIA, 2025).

A competição associada ao conjunto Nexar avalia os modelos por meio da métrica *Average Precision* em diferentes intervalos anteriores ao acidente, incluindo 500, 1.000 e 1.500 milissegundos antes do evento. Essa estratégia enfatiza a necessidade de produzir previsões simultaneamente antecipadas e confiáveis, aproximando o problema experimental das exigências encontradas em aplicações reais de segurança veicular (MOURA; ZHU; ZVITIA, 2025).

Diante desse cenário, este trabalho propõe o desenvolvimento e a avaliação de uma pipeline aplicada de inteligência artificial para a previsão antecipada de colisões e quase-colisões a partir de vídeos de *dashcam*. A abordagem contempla o processamento dos vídeos, a extração de representações visuais, a construção de modelos com diferentes níveis de complexidade e a análise temporal das probabilidades estimadas.

Inicialmente, é desenvolvido um modelo de referência baseado na extração de características visuais por meio de uma rede neural convolucional pré-treinada. As características obtidas a partir dos frames são agregadas e utilizadas por um classificador responsável por estimar a probabilidade de colisão. Esse modelo estabelece um baseline experimental e permite avaliar a capacidade das informações visuais estáticas ou agregadas de representar situações de risco.

Posteriormente, são investigados modelos capazes de representar explicitamente a dependência temporal entre os frames, como arquiteturas que combinam redes neurais convolucionais com redes LSTM ou GRU. A comparação entre essas abordagens permite analisar em que medida a incorporação direta da dinâmica temporal contribui para a antecipação das colisões em relação a um modelo convolucional com pós-processamento temporal simples. Arquiteturas mais complexas para processamento de vídeo, como redes convolucionais tridimensionais e modelos baseados em mecanismos de atenção e Transformers, permanecem como extensões futuras deste trabalho.

Além da avaliação classificatória convencional, este trabalho considera a evolução da probabilidade de colisão ao longo do vídeo e a antecedência com que o risco é identificado. Embora a competição associada ao conjunto Nexar utilize *Average Precision* em janelas temporais anteriores ao evento, a avaliação experimental conduzida neste trabalho prioriza métricas diretamente relacionadas ao comportamento de alerta, como precisão, revocação, F1-score, área sob a curva ROC, taxa de falsos alarmes, taxa de eventos perdidos e erro temporal médio do alerta. Também é conduzida uma análise dos falsos positivos e falsos negativos, buscando identificar condições visuais e temporais associadas aos principais erros dos modelos.

Como parte da solução de engenharia, também é proposta uma interface web interativa para visualização dos resultados. A interface permite reproduzir o vídeo analisado, acompanhar a evolução temporal da probabilidade de colisão, identificar frames críticos, comparar a classe real com a classe predita e examinar os erros produzidos pelos diferentes modelos. Embora essa interface não constitua o elemento central da contribuição científica, ela complementa a metodologia ao facilitar a interpretação dos resultados e a análise do comportamento dos modelos, além de aproximar a pipeline experimental de um produto de análise de risco baseado em vídeos de *dashcam*.

As principais contribuições propostas por este trabalho são:

a) o desenvolvimento de uma pipeline reprodutível para processamento, análise e modelagem de vídeos veiculares;

b) a construção de um baseline baseado em características visuais extraídas de múltiplos frames;

c) a comparação entre abordagens baseadas em informações visuais estáticas, agregadas e temporais;

d) a avaliação dos modelos considerando tanto o desempenho classificatório quanto a antecedência das previsões;

e) a análise dos principais padrões associados aos falsos positivos e falsos negativos; e

f) o desenvolvimento de uma interface web interativa para visualização da evolução temporal do risco e interpretação dos resultados.

Com isso, busca-se contribuir para o desenvolvimento de soluções de inteligência artificial aplicadas à segurança veicular, aproximando a experimentação com modelos de visão computacional das necessidades de antecipação, interpretação e reprodutibilidade encontradas em sistemas reais de apoio à condução.

# Referências

BAO, Wentao; YU, Qi; KONG, Yu. Uncertainty-based traffic accident anticipation with spatio-temporal relational learning. In: **ACM International Conference on Multimedia**, 28., 2020, Seattle. *Proceedings [...]*. New York: Association for Computing Machinery, 2020. p. 2682–2690. DOI: 10.1145/3394171.3413827.

CHAN, Fu-Hsiang; CHEN, Yu-Ting; XIANG, Yu; SUN, Min. Anticipating accidents in dashcam videos. In: LAI, Shang-Hong et al. (ed.). *Computer Vision — ACCV 2016*. Cham: Springer, 2017. p. 136–153. Lecture Notes in Computer Science, v. 10114. DOI: 10.1007/978-3-319-54190-7_9.

MOURA, Daniel C.; ZHU, Shizhan; ZVITIA, Orly. Nexar Dashcam Collision Prediction Dataset and Challenge. In: **IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops**, 2025. *Proceedings [...]*. Nashville: IEEE/CVF, 2025.

WORLD HEALTH ORGANIZATION. *Global status report on road safety 2023*. Geneva: World Health Organization, 2023. ISBN 978-92-4-008651-7.

