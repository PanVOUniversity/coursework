import React from 'react';
import { motion } from 'framer-motion';
import { FaHome, FaKey, FaChartBar, FaFileContract } from 'react-icons/fa';
import './Services.css';

const Services = () => {
  const services = [
    {
      icon: FaHome,
      title: 'Продажа недвижимости',
      description: 'Поможем продать вашу недвижимость по максимальной цене'
    },
    {
      icon: FaKey,
      title: 'Аренда',
      description: 'Найдем идеальную аренду для ваших потребностей'
    },
    {
      icon: FaChartBar,
      title: 'Инвестиции',
      description: 'Консультации по инвестициям в недвижимость'
    },
    {
      icon: FaFileContract,
      title: 'Юридическое сопровождение',
      description: 'Полное юридическое сопровождение сделок'
    }
  ];

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
            Полный спектр услуг в сфере недвижимости Москвы
          </p>
        </motion.div>

        <div className="services-grid">
          {services.map((service, index) => (
            <motion.div
              key={index}
              className="service-card"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              viewport={{ once: true }}
              whileHover={{ y: -10 }}
            >
              <div className="service-icon-container">
                <motion.div
                  className="service-icon"
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
        </div>
      </div>
    </section>
  );
};

export default Services;
