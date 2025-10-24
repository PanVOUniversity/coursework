import React from 'react';
import { motion } from 'framer-motion';
import { FaRocket, FaUsers, FaAward, FaLightbulb } from 'react-icons/fa';
import './About.css';

const About = () => {
  const stats = [
    {
      icon: FaRocket,
      number: 200,
      suffix: '+',
      label: 'Проектов',
      color: '#7c3aed'
    },
    {
      icon: FaUsers,
      number: 50,
      suffix: '+',
      label: 'Клиентов',
      color: '#a855f7'
    },
    {
      icon: FaAward,
      number: 5,
      suffix: '',
      label: 'Лет опыта',
      color: '#c084fc'
    },
    {
      icon: FaLightbulb,
      number: 150,
      suffix: '%',
      label: 'Рост продаж',
      color: '#e9d5ff'
    }
  ];

  return (
    <section id="about" className="about">
      <div className="container">
        <motion.div
          className="about-content"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="about-text">
            <div className="about-badge">
              <span>🚀 О нас</span>
            </div>
            
            <h2 className="about-title">
              <span className="gradient-text">Creative Agency</span>
              <br />
              Ваш партнер в мире креативного маркетинга
            </h2>
            
            <div className="about-description">
              <p>
                Мы - креативное маркетинговое агентство с более чем 5-летним опытом работы. 
                Наша команда профессионалов помогает брендам выделяться на рынке и привлекать новых клиентов.
              </p>
              <p>
                Мы специализируемся на создании уникальных маркетинговых решений, которые не только 
                привлекают внимание, но и приносят реальные результаты для бизнеса.
              </p>
            </div>
          </div>

          <div className="about-visual">
            <div className="about-image-container">
              <img
                src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
                alt="Creative Team"
                className="about-image"
              />
            </div>

            <div className="stats-container">
              {stats.map((stat, index) => (
                <motion.div
                  key={index}
                  className="stat-card"
                  style={{ backgroundColor: stat.color }}
                  whileHover={{ scale: 1.05 }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="stat-icon">
                    <stat.icon />
                  </div>
                  <div className="stat-number">
                    {stat.number}{stat.suffix}
                  </div>
                  <div className="stat-label">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default About;
