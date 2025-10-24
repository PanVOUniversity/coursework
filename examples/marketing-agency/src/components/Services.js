import React from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaPalette, FaBullhorn, FaChartBar, FaMobile, FaSearch, FaVideo } from 'react-icons/fa';
import './Services.css';

const Services = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const services = [
    {
      icon: FaPalette,
      title: 'Брендинг',
      description: 'Создание уникального визуального образа вашего бренда',
      color: '#7c3aed'
    },
    {
      icon: FaBullhorn,
      title: 'Реклама',
      description: 'Эффективные рекламные кампании для привлечения клиентов',
      color: '#a855f7'
    },
    {
      icon: FaChartBar,
      title: 'Аналитика',
      description: 'Глубокий анализ данных для оптимизации маркетинга',
      color: '#c084fc'
    },
    {
      icon: FaMobile,
      title: 'SMM',
      description: 'Продвижение в социальных сетях и управление сообществами',
      color: '#e9d5ff'
    },
    {
      icon: FaSearch,
      title: 'SEO',
      description: 'Оптимизация сайта для поисковых систем',
      color: '#7c3aed'
    },
    {
      icon: FaVideo,
      title: 'Видеопродакшн',
      description: 'Создание качественного видеоконтента',
      color: '#a855f7'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="services" className="services">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Наши услуги</h2>
          <p className="section-subtitle">
            Полный спектр маркетинговых услуг для развития вашего бизнеса
          </p>
        </motion.div>

        <motion.div
          className="services-grid"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {services.map((service, index) => (
            <motion.div
              key={index}
              className="service-card"
              variants={itemVariants}
              whileHover={{ 
                y: -10,
                transition: { duration: 0.3 }
              }}
            >
              <div className="service-icon-container">
                <motion.div
                  className="service-icon"
                  style={{ backgroundColor: service.color }}
                  whileHover={{ 
                    scale: 1.1,
                    rotate: 5,
                    transition: { duration: 0.3 }
                  }}
                >
                  <service.icon />
                </motion.div>
              </div>
              
              <h3 className="service-title">{service.title}</h3>
              <p className="service-description">{service.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Services;
