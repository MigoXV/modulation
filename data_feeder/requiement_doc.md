# 信号发生器需求文档

应该在wave_generator.py中定义wave_generator类，保证代码的高效性和易用性

## 配置文件

### generator_config.yml

保存采样率、每个数据生成的总长度，为了随机相位实际选择的长度；载波峰峰值，频率；各调制波形的参数，如 $m_a$ 的上下限

### waves.jsonl

保存每种调制后波形的生成个数，以及每种调制信号的频率。

## wave_generator类

信号生成器类，可以一键产生配置文件中声明的所有种类的信号

### 属性

- 配置
- 波形字典
- waves张量

### init

传入一个omegaconf对象config，解析后初始化自身属性

### gene_waves方法

该方法用于一键产生所有的信号。算法为，分别产生6种不同的信号，将这些信号保存到自身的waves张量内