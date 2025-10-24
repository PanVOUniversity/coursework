import React from 'react';
import { motion } from 'framer-motion';
import { FaExternalLinkAlt, FaEye } from 'react-icons/fa';
import './Portfolio.css';

const Portfolio = () => {
  const projects = [
    {
      id: 1,
      title: 'Rebranding для Tech Startup',
      category: 'Брендинг',
      image: 'https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Полный ребрендинг технологического стартапа'
    },
    {
      id: 2,
      title: 'SMM кампания для ресторана',
      category: 'SMM',
      image: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Продвижение ресторана в социальных сетях'
    },
    {
      id: 3,
      title: 'Видео для модного бренда',
      category: 'Видеопродакшн',
      image: 'https://images.unsplash.com/photo-1553877522-43269d4ea984?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Создание рекламного видео для модного бренда'
    },
    {
      id: 4,
      title: 'SEO для интернет-магазина',
      category: 'SEO',
      image: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Оптимизация сайта интернет-магазина'
    },
    {
      id: 5,
      title: 'Реклама для фитнес-центра',
      category: 'Реклама',
      image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Рекламная кампания для фитнес-центра'
    },
    {
      id: 6,
      title: 'Аналитика для e-commerce',
      category: 'Аналитика',
      image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      description: 'Анализ данных интернет-магазина'
    }
  ];

  return (
    <section id="portfolio" className="portfolio">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Наши работы</h2>
          <p className="section-subtitle">
            Примеры успешных проектов для наших клиентов
          </p>
        </motion.div>

        <div className="portfolio-grid">
          {projects.map((project, index) => (
            <motion.div
              key={project.id}
              className="portfolio-item"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ y: -10 }}
            >
              <div className="portfolio-image-container">
                <img
                  src={project.image}
                  alt={project.title}
                  className="portfolio-image"
                />
                <div className="portfolio-overlay">
                  <div className="portfolio-actions">
                    <motion.button
                      className="portfolio-btn"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <FaEye />
                    </motion.button>
                    <motion.button
                      className="portfolio-btn"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <FaExternalLinkAlt />
                    </motion.button>
                  </div>
                </div>
                <div className="portfolio-category">{project.category}</div>
              </div>

              <div className="portfolio-content">
                <h3 className="portfolio-title">{project.title}</h3>
                <p className="portfolio-description">{project.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Portfolio;
