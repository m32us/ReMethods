# Layer-wise Relevance Propagation in PyTorch with Mixed-Precision Training

Implementation of unsupervised Layer-wise Relevance Propagation (LRP, Bach et al., Montavon et al.) in PyTorch with Mixed-Precision Training for VGG networks from scratch. [This tutorial](https://git.tu-berlin.de/gmontavon/lrp-tutorial) served as a starting point. In this implementation, we provide study about Layer-wise Relevance Propagation in our master class (HCMUS Master Course - Research Methodologies), and an framework that easy to understand for PyTorch users.

In this repository, we base on a novel relevance propagation filter to this implementation resulting in much crisper heatmaps that could be found in [Fischer, Kai's blog](https://kaifishr.github.io/). We provide two strategies for training your network, normal way like you learned in your school, and mixed precision training. Furthermore, we use layer-wise propagation helps us to identify input features that were relevant for network’s classification decision. Almost source code you can see in [Layer-wise Relevance Propagation in PyTorch](https://github.com/kaifishr/PyTorchRelevancePropagation) by Fischer, Kai. For producing experiment results, we use GTX 1050 Nvidia graphics card. The FPS can be improved with other stronger graphics card.

Useful resources:

[1] [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

[2] [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10)

[3] [LRP tutorial](https://git.tu-berlin.de/gmontavon/lrp-tutorial)

## General information

- Trần Xuân Lộc (22C11064)
- Nguyễn Bảo Long (22C11065)
- Lê Nhựt Nam (22C11067)

## How to run project

Read the README in source codes folder.

## LICENSE

[MIT LICENSE](./LICENSE)

## Citation

```
@misc{blogpost,
  title={Layer-wise Relevance Propagation for PyTorch},
  author={Fischer, Kai},
  howpublished={\url{https://github.com/kaifishr/PyTorchRelevancePropagation}},
  year={2021}
}
```

```
@misc{blogpost,
  title={Improved Layer-wise Relevance Propagation for PyTorch with Mixed-Precision Training},
  author={Tran X. Loc, Nguyen B. Long, Le N. Nam},
  howpublished={\url{https://github.com/m32us/ReMethods}},
  year={2023}
}
```

## Acknowledgements

Thanks to my friends, [Tran Xuan Loc](https://github.com/stark4079), [Nguyen Bao Long](https://github.com/baolongnguyenmac)

## References

[1] M. T. Ribeiro, S. Singh, and C. Guestrin, “" why should i trust you?" explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135–1144, 2016.

[2] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[3] D. Smilkov, N. Thorat, B. Kim, F. Viégas, and M. Wattenberg, “Smoothgrad: removing noise by adding noise,” arXiv preprint arXiv:1706.03825, 2017.

[4] A. Shrikumar, P. Greenside, A. Shcherbina, and A. Kundaje, “Not just a blackbox: Learning important features through propagating activation differences,” arXiv preprint arXiv:1605.01713, 2016.

[5] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. M ̈uller, and W. Samek, “On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation,” PloS one, vol. 10, no. 7, p. e0130140, 2015.

[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, pp. 618–626, 2017.

[7] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2921–2929, 2016.

[8] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 782–791, 2021.

[9] B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, F. Viegas, and R. Sayres, “Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav).(2017),” arXiv preprint arXiv:1711.11279, 2017.

[10] S. Wachter, B. Mittelstadt, and C. Russell, “Counterfactual explanations without opening the black box: Automated decisions and the gdpr,” Harv. JL & Tech., vol. 31, p. 841, 2017.

[11] A. Krizhevsky, G. Hinton, et al., “Learning multiple layers of features from tiny images,” 2009.

[12] W. Samek, A. Binder, G. Montavon, S. Bach, and K. M ̈uller, “Evaluating the visualization of what a deep neural network has learned. arxiv,” arXiv preprint arXiv:1509.06321, 2015.

[13] A. Binder, G. Montavon, S. Lapuschkin, K.-R. M ̈uller, and W. Samek, “Layer-wise relevance propagation for neural networks with local renormalization layers,” in Artificial Neural Networks and Machine Learning–ICANN 2016: 25th International Conference on Artificial Neural Networks, Barcelona, Spain, September 6-9, 2016, Proceedings, Part II 25, pp. 63–71, Springer, 2016.

[14] G. Montavon, A. Binder, S. Lapuschkin, W. Samek, and K.-R. M ̈uller, “Layer-wise relevance propagation: an overview,” Explainable AI: interpreting, explaining and visualizing deep learning, pp. 193–209, 2019.

[15] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi, “A survey of methods for explaining black box models,” ACM computing surveys (CSUR), vol. 51, no. 5, pp. 1–42, 2018.

[16] W. Ding, M. Abdel-Basset, H. Hawash, and A. M. Ali, “Explainability of artificial intelligence methods, applications and challenges: A comprehensive survey,” Information Sciences, 2022.

[17] G. Ras, N. Xie, M. Van Gerven, and D. Doran, “Explainable deep learning: A field guide for the uninitiated,” Journal of Artificial Intelligence Research, vol. 73, pp. 329–397, 2022.

[18] X. Li, H. Xiong, X. Li, X. Wu, X. Zhang, J. Liu, J. Bian, and D. Dou, “Interpretable deep learning: Interpretation, interpretability, trustworthiness, and beyond,” Knowledge and Information Systems, vol. 64, no. 12, pp. 3197–3234,
2022.

[19] T. Speith, “A review of taxonomies of explainable artificial intelligence (xai) methods,” in 2022 ACM Conference on Fairness, Accountability, and Transparency, pp. 2239–2250, 2022.
