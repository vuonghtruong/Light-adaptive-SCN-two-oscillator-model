import figure1,figures34,architecture_figures,supplement,technical_appendix,memory_timescale,simple_coupling_rule,graphical_abstract

def render_figures():
    figure1.main()
    figures34.main3();figures34.main4()
    architecture_figures.main2();architecture_figures.s3()
    supplement.s1();supplement.s2();supplement.s4()
    memory_timescale.render();simple_coupling_rule.render()
    technical_appendix.main()
    graphical_abstract.main()
    print('All four main figures, S1-S6, the technical appendix and graphical abstract rendered.',flush=True)

if __name__=='__main__':
    from reproduce_pipeline import main
    main()
