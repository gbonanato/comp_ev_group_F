# Computação Evolucionária Grupo F

Este repositório foi criado para a resolução dos problemas da disciplina de Computação Evolucionária do PPGEE UFMG 26/1.

## Como usar o repositório:

O repositório está estruturado como uma biblioteca para o trabalho prático 1 (TP1). Para utilizar o pacote é recomendada criação de ambiente virtual utilizando o comando:

```python
python -m venv .venv
```
*Obs: O nome ".venv" pode ser parametrizado pelo usuário da forma que preferir. Se trata apenas do nome do ambiente virtual a ser criado*

Certifique que a versão do python do ambiente virtual seja alguma distribuição do python 3.13 (de preferência a mais recente). Isso garante um comportamento mais estável do pacote e com performance semelhante àquela utilizada no desenvolvimento local. Caso utilize a distribuição conda, o ambiente pode ser criado na versão 3.13 com o seguinte comando:

```python
conda create -n .venv python=3.13
```

Após criado, ativar o ambiente virtual com:

```cmd
.venv\Scripts\activate
```

Ou se usar a distribuição conda:

```python
conda activate .venv
```

Uma vez com o ambiente virtual ativado, as dependências de desenvolvimento do projeto podem ser instaladas em modo editável com o comando:

```cmd
pip install -e .
```

A instalação em modo editável faz com que quaisquer atualizações do código base sejam refletidas imediatamente na importação dos pacote, sem que seja necessário reeinstalar novamente.



## Trabalho 1

O trabalho 1 consiste em dois problemas de computação evolucionária disponíveis [aqui](docs\TP1\OrientacoesTrabalhoPratico1.pdf).
