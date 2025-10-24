import React from 'react';
import { motion } from 'framer-motion';
import { FaCut, FaPaintBrush, FaSpa, FaFemale } from 'react-icons/fa';
import './styles.css';

const Services = () => {
  const services = [
    { icon: FaCut, title: 'Стрижки', description: 'Модные женские и мужские стрижки', color: '#ec4899' },
    { icon: FaPaintBrush, title: 'Окрашивание', description: 'Профессиональное окрашивание волос', color: '#f472b6' },
    { icon: FaSpa, title: 'Уход', description: 'Процедуры по уходу за волосами', color: '#fbcfe8' },
    { icon: FaFemale, title: 'Укладка', description: 'Праздничные и повседневные укладки', color: '#be185d' }
  ];

  return (
    <section id="services" className="services">
      <div className="container">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="section-title">Наши услуги</h2>
        </motion.div>

        <div className="services-grid">
          {services.map((service, index) => (
            <motion.div key={index} className="service-card" initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.2 }} 
              viewport={{ once: true }} whileHover={{ y: -10 }}>
              <div className="service-icon" style={{ backgroundColor: service.color }}>
                <service.icon />
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
